//! wgpu harness: uploads line soup and entries, dispatches one merge variant, reads back.

use wgpu::util::DeviceExt;

use crate::tiler::{Entry, PIXELS, TILE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Phase 2 only, no merge (diagnostic floor).
    ComputeOnly,
    OneLine,
    TwoLines,
}

#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub mode: Mode,
    /// Profile level for `Mode::ComputeOnly` (see shader/merge0.wgsl); ignored otherwise.
    pub profile: u32,
    /// Evaluate every cell with the full area formula (no coset classification / FIFO).
    pub dense: bool,
    pub wg_size: u32,
    pub chunk: u32,
}

impl Variant {
    pub fn entries_per_wg(&self) -> usize {
        (self.wg_size as usize) * if self.mode == Mode::TwoLines { 2 } else { 1 }
    }

    pub fn name(&self) -> String {
        let mode = match self.mode {
            Mode::ComputeOnly => "computeonly",
            Mode::OneLine => "oneline",
            Mode::TwoLines => "twoline",
        };
        let dense = if self.dense { "_dense" } else { "" };
        if self.mode == Mode::ComputeOnly {
            format!("{mode}{}_wg{}_chunk{}{dense}", self.profile, self.wg_size, self.chunk)
        } else {
            format!("{mode}_wg{}_chunk{}{dense}", self.wg_size, self.chunk)
        }
    }

    pub fn shader_source(&self) -> String {
        let header = format!(
            "const TILE: u32 = {}u;\nconst WG_SIZE: u32 = {}u;\nconst CHUNK: u32 = {}u;\nconst PROFILE: u32 = {}u;\nconst DENSE: u32 = {}u;\n",
            TILE, self.wg_size, self.chunk, self.profile, self.dense as u32
        );
        let body = match self.mode {
            Mode::ComputeOnly => include_str!("../shader/merge0.wgsl"),
            Mode::OneLine => include_str!("../shader/merge1.wgsl"),
            Mode::TwoLines => include_str!("../shader/merge2.wgsl"),
        };
        format!("{header}{}\n{body}", include_str!("../shader/phase2.wgsl"))
    }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub has_timestamps: bool,
}

pub struct GpuResult {
    pub alpha: Vec<u32>,
    pub partials: Vec<i32>,
    /// Per-iteration GPU times in milliseconds, if timestamps are supported.
    pub times_ms: Vec<f64>,
}

impl Gpu {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("no adapter");
        let info = adapter.get_info();
        println!("adapter: {} ({:?})", info.name, info.backend);
        let has_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let mut features = wgpu::Features::empty();
        if has_timestamps {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("merge harness"),
            required_features: features,
            ..Default::default()
        }))
        .expect("no device");
        Self {
            device,
            queue,
            has_timestamps,
        }
    }

    pub fn run(
        &self,
        variant: Variant,
        lines: &[[f32; 4]],
        entries: &[Entry],
        path_flags: &[u32],
        n_alpha: u32,
        n_partials: u32,
        iterations: u32,
    ) -> GpuResult {
        let device = &self.device;
        assert_eq!(entries.len() % variant.entries_per_wg(), 0);
        let n_wg = (entries.len() / variant.entries_per_wg()) as u32;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&variant.name()),
            source: wgpu::ShaderSource::Wgsl(variant.shader_source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&variant.name()),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let storage = wgpu::BufferUsages::STORAGE;
        let lines_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lines"),
            contents: bytemuck::cast_slice(lines),
            usage: storage,
        });
        let entries_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("entries"),
            contents: bytemuck::cast_slice(entries),
            usage: storage,
        });
        let flags_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("path_flags"),
            contents: bytemuck::cast_slice(if path_flags.is_empty() { &[0u32] } else { path_flags }),
            usage: storage,
        });
        let alpha_size = (n_alpha.max(1) as u64) * PIXELS as u64;
        let partials_size = (n_partials.max(1) as u64) * PIXELS as u64 * 4;
        let alpha_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alpha"),
            size: alpha_size,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("partials"),
            size: partials_size,
            usage: storage | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: alpha_size + partials_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lines_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: entries_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: alpha_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partials_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: flags_buf.as_entire_binding(),
                },
            ],
        });

        let n_queries = 2 * iterations;
        let (query_set, query_buf, query_readback) = if self.has_timestamps {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: n_queries,
            });
            let qb = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve"),
                size: 8 * n_queries as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let qr = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query readback"),
                size: 8 * n_queries as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            (Some(qs), Some(qb), Some(qr))
        } else {
            (None, None, None)
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // Untimed warm-up so the GPU clock has ramped before the timed passes.
        for _ in 0..iterations.max(50) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmup"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        for it in 0..iterations {
            let timestamp_writes = query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(2 * it),
                end_of_pass_write_index: Some(2 * it + 1),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge"),
                timestamp_writes,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(n_wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&alpha_buf, 0, &readback, 0, alpha_size);
        encoder.copy_buffer_to_buffer(&partials_buf, 0, &readback, alpha_size, partials_size);
        if let (Some(qs), Some(qb), Some(qr)) = (&query_set, &query_buf, &query_readback) {
            encoder.resolve_query_set(qs, 0..n_queries, qb, 0);
            encoder.copy_buffer_to_buffer(qb, 0, qr, 0, 8 * n_queries as u64);
        }
        self.queue.submit([encoder.finish()]);

        readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        if let Some(qr) = &query_readback {
            qr.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        }
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

        let data = readback.slice(..).get_mapped_range().expect("mapped range");
        let alpha: Vec<u32> = bytemuck::cast_slice(&data[..alpha_size as usize]).to_vec();
        let partials: Vec<i32> = bytemuck::cast_slice(&data[alpha_size as usize..]).to_vec();
        drop(data);
        readback.unmap();

        let mut times_ms = Vec::new();
        if let Some(qr) = &query_readback {
            let period = self.queue.get_timestamp_period() as f64;
            let data = qr.slice(..).get_mapped_range().expect("mapped range");
            let ts: &[u64] = bytemuck::cast_slice(&data);
            for it in 0..iterations as usize {
                let dt = ts[2 * it + 1].wrapping_sub(ts[2 * it]);
                times_ms.push(dt as f64 * period * 1e-6);
            }
            drop(data);
            qr.unmap();
        }

        GpuResult {
            alpha,
            partials,
            times_ms,
        }
    }
}
