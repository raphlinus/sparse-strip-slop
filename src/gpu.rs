//! wgpu harness: uploads line soup and entries, dispatches one merge variant, reads back.

use wgpu::util::DeviceExt;

use crate::tiler::{Entry, PIXELS, TILE};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Phase 2 only, no merge (diagnostic floor).
    ComputeOnly,
    OneLine,
    TwoLines,
    /// Self-contained stripped-down shader (one line per thread, sparse, chunk 1).
    Simple,
    /// 8x8 entries expanded to 4x4 quadrant items with load balancing (TILE must be 8).
    Quadrant,
    /// MSAA 16 payload, one entry per thread, chunk 1; `cell_mode` selects the per-cell method.
    Msaa,
    /// MSAA 16 payload, two entries per thread (case table).
    Msaa2,
    /// MSAA 16 payload, 8x8 entries merged as load-balanced 4x4 quadrants.
    MsaaQuadrant,
    /// MSAA 16 payload, one 4x4 quadrant item per thread from a host-expanded item list.
    MsaaFlat,
}

#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub mode: Mode,
    /// Profile level for `Mode::ComputeOnly` (see shader/merge0.wgsl); ignored otherwise.
    pub profile: u32,
    /// Evaluate every cell with the full area formula (no coset classification / FIFO).
    pub dense: bool,
    /// Quadrant mode: item slots per thread (3 => entries per workgroup = threads,
    /// 2 => entries = threads / 2).
    pub quad_slots: u32,
    /// Msaa mode: 0 inline LUT, 1 coset FIFO of LUT masks, 2 direct 16-sample evaluation.
    pub cell_mode: u32,
    pub wg_size: u32,
    pub chunk: u32,
}

impl Variant {
    /// Entries per piece before a new output slot is needed (the quadrant variant only
    /// splits at workgroup boundaries).
    pub fn piece_size(&self) -> usize {
        if matches!(self.mode, Mode::Quadrant | Mode::MsaaQuadrant | Mode::MsaaFlat) { usize::MAX } else { crate::tiler::PIECE }
    }

    /// Pixel (x, y) -> index within a run's output tile.
    pub fn layout_index(&self, x: u32, y: u32) -> usize {
        if matches!(self.mode, Mode::Quadrant | Mode::MsaaQuadrant | Mode::MsaaFlat) {
            crate::reference::quadrant_major_index(x, y)
        } else {
            crate::reference::coset_major_index(x, y)
        }
    }

    pub fn is_msaa(&self) -> bool {
        matches!(self.mode, Mode::Msaa | Mode::Msaa2 | Mode::MsaaQuadrant | Mode::MsaaFlat)
    }

    pub fn entries_per_wg(&self) -> usize {
        match self.mode {
            Mode::TwoLines | Mode::Msaa2 => self.wg_size as usize * 2,
            Mode::Quadrant => if self.quad_slots == 2 { self.wg_size as usize / 2 } else { self.wg_size as usize },
            _ => self.wg_size as usize,
        }
    }

    pub fn name(&self) -> String {
        let mode = match self.mode {
            Mode::ComputeOnly => "computeonly",
            Mode::OneLine => "oneline",
            Mode::TwoLines => "twoline",
            Mode::Simple => "simple",
            Mode::Quadrant => "quadrant",
            Mode::Msaa => "msaa",
            Mode::Msaa2 => "msaa2",
            Mode::MsaaQuadrant => "msaaquad",
            Mode::MsaaFlat => "msaaflat",
        };
        let dense = if self.dense { "_dense" } else { "" };
        if self.mode == Mode::Quadrant {
            return format!("{mode}_wg{}_e{}", self.wg_size, self.entries_per_wg());
        }
        if self.is_msaa() {
            let cm = ["inlinelut", "fifolut", "direct", "alwayslut"][self.cell_mode as usize];
            return format!("{mode}_{cm}_wg{}", self.wg_size);
        }
        if self.mode == Mode::Simple {
            return format!("{mode}_wg{}", self.wg_size);
        }
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
        if self.is_msaa() {
            let mprof: u32 = std::env::var("MPROF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let mabl: u32 = std::env::var("MABL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let qsetup = include_str!("../shader/msaa_quadrant_setup.wgsl");
            let body = match self.mode {
                Mode::Msaa => include_str!("../shader/msaa1.wgsl").to_string(),
                Mode::Msaa2 => include_str!("../shader/msaa2.wgsl").to_string(),
                Mode::MsaaQuadrant => format!("{qsetup}\n{}", include_str!("../shader/msaa_quadrant.wgsl")),
                _ => format!("{qsetup}\n{}", include_str!("../shader/msaa_flat.wgsl")),
            };
            return format!(
                "const TILE: u32 = {}u;\nconst WG_SIZE: u32 = {}u;\nconst CELL_MODE: u32 = {}u;\nconst MPROF: u32 = {}u;\nconst MABL: u32 = {}u;\n{}\n{body}",
                TILE,
                self.wg_size,
                self.cell_mode,
                mprof,
                mabl,
                include_str!("../shader/msaa_common.wgsl")
            );
        }
        if self.mode == Mode::Simple || self.mode == Mode::Quadrant {
            let body = if self.mode == Mode::Simple {
                include_str!("../shader/merge_simple.wgsl")
            } else {
                include_str!("../shader/merge_quadrant.wgsl")
            };
            let qprofile: u32 = std::env::var("QPROFILE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            return format!(
                "const TILE: u32 = {}u;\nconst WG_SIZE: u32 = {}u;\nconst ENTRIES: u32 = {}u;\nconst QPROFILE: u32 = {}u;\n{body}",
                TILE,
                self.wg_size,
                self.entries_per_wg(),
                qprofile
            );
        }
        let body = match self.mode {
            Mode::ComputeOnly => include_str!("../shader/merge0.wgsl"),
            Mode::OneLine => include_str!("../shader/merge1.wgsl"),
            Mode::TwoLines => include_str!("../shader/merge2.wgsl"),
            Mode::Simple | Mode::Quadrant | Mode::Msaa | Mode::Msaa2 | Mode::MsaaQuadrant | Mode::MsaaFlat => unreachable!(),
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
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            eprintln!("wgpu uncaptured error: {e}");
        }));
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
        lut: Option<&[u32]>,
        items: Option<&[[u32; 4]]>,
    ) -> GpuResult {
        let device = &self.device;
        assert_eq!(entries.len() % variant.entries_per_wg(), 0);
        let n_wg = match items {
            Some(it) => {
                assert_eq!(it.len() % variant.wg_size as usize, 0);
                (it.len() / variant.wg_size as usize) as u32
            }
            None => (entries.len() / variant.entries_per_wg()) as u32,
        };

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
        // Area: 1 byte per pixel, 4 bytes per partial sample. MSAA: 2 bytes per pixel
        // (16-bit mask), 16 x 4 bytes per partial pixel.
        let (alpha_size, partials_size) = if variant.is_msaa() {
            ((n_alpha.max(1) as u64) * PIXELS as u64 * 2, (n_partials.max(1) as u64) * PIXELS as u64 * 16)
        } else {
            ((n_alpha.max(1) as u64) * PIXELS as u64, (n_partials.max(1) as u64) * PIXELS as u64 * 4)
        };
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

        let lut_buf = lut.map(|l| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mask_lut"),
                contents: bytemuck::cast_slice(l),
                usage: storage,
            })
        });
        let mut bind_entries = vec![
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
        ];
        if let Some(lb) = &lut_buf {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: lb.as_entire_binding(),
            });
        }
        let items_buf = items.map(|it| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("items"),
                contents: bytemuck::cast_slice(it),
                usage: storage,
            })
        });
        if let Some(ib) = &items_buf {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: ib.as_entire_binding(),
            });
        }
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bind_entries,
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

/// Result of the GPU expansion + merge pipeline.
pub struct ExpandedResult {
    pub masks: Vec<u32>,
    pub partials: Vec<i32>,
    pub entries4: Vec<Entry>,
    /// run_info[run * 4 + q]: record index of an untouched quadrant's uniform record,
    /// NONE (0xffffffff) for touched quadrants, NONE - 1 for untouched in resolved runs.
    pub run_info: Vec<u32>,
    pub n_runs: u32,
    pub n_items: u32,
    /// Per iteration: [e1, e2, e3, e5, e4, merge] milliseconds.
    pub times_ms: Vec<[f64; 6]>,
}

impl Gpu {
    /// Expand sorted 8x8 entries into 4x4 quadrant entries on the GPU, then merge.
    /// `n_items`, `n_runs_total` and `n_records` are host-side upper bounds for buffer sizes.
    pub fn run_expanded(
        &self,
        wg_size: u32,
        cell_mode: u32,
        lines: &[[f32; 4]],
        entries8: &[Entry],
        path_flags: &[u32],
        lut: &[u32],
        n_items: usize,
        n_runs_total: usize,
        n_records: usize,
        dummy_line: u32,
        iterations: u32,
    ) -> ExpandedResult {
        let device = &self.device;
        let n = entries8.len();
        assert_eq!(n % wg_size as usize, 0);
        let n_wg = (n / wg_size as usize) as u32;
        let n4 = n_items.div_ceil(wg_size as usize) * wg_size as usize;
        let n4 = n4.max(wg_size as usize);
        let n_mask_slots = n_runs_total + 2; // + dummy slot for the sentinel run and padding

        // Scratch layout (u32 words).
        let off_base = 0usize;
        let off_cnt = n;
        let off_fill = 5 * n;
        let off_delta = 9 * n;
        let off_meta = 13 * n;
        let off_rinfo = 17 * n;
        let off_ctr = 21 * n;
        let off_flags = 21 * n + 4;
        let scratch_len = off_flags + path_flags.len().max(1);
        let header = format!(
            "const WG_SIZE: u32 = {wg_size}u;\nconst N: u32 = {n}u;\nconst N_WG: u32 = {n_wg}u;\n\
             const OFF_BASE: u32 = {off_base}u;\nconst OFF_CNT: u32 = {off_cnt}u;\nconst OFF_FILL: u32 = {off_fill}u;\n\
             const OFF_DELTA: u32 = {off_delta}u;\nconst OFF_META: u32 = {off_meta}u;\nconst OFF_RINFO: u32 = {off_rinfo}u;\n\
             const OFF_CTR: u32 = {off_ctr}u;\nconst OFF_FLAGS: u32 = {off_flags}u;\n"
        );
        let expand_src = format!("{header}{}", include_str!("../shader/expand.wgsl"));
        let expand_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("expand"),
            source: wgpu::ShaderSource::Wgsl(expand_src.into()),
        });
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8)
            .map(|b| wgpu::BindGroupLayoutEntry {
                binding: b,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("expand"),
            entries: &bgl_entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("expand"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &expand_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let p_e1 = make("e1_map");
        let p_e2 = make("e2_scan");
        let p_e3 = make("e3_runs");
        let p_e5 = make("e5_scatter");
        let p_e4 = make("e4_fixup");

        let merge_src = format!(
            "const TILE: u32 = 8u;\nconst WG_SIZE: u32 = {wg_size}u;\nconst CELL_MODE: u32 = {cell_mode}u;\nconst MPROF: u32 = 0u;\nconst MABL: u32 = 0u;\n{}\n{}\n{}",
            include_str!("../shader/msaa_common.wgsl"),
            include_str!("../shader/msaa_quadrant_setup.wgsl"),
            include_str!("../shader/msaa_items.wgsl")
        );
        let merge_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("msaa_items"),
            source: wgpu::ShaderSource::Wgsl(merge_src.into()),
        });
        let p_merge = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("msaa_items"),
            layout: None,
            module: &merge_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let storage = wgpu::BufferUsages::STORAGE;
        let rw = storage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let mk_init = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: bytes, usage })
        };
        let lines_buf = mk_init("lines", bytemuck::cast_slice(lines), storage);
        let entries8_buf = mk_init("entries8", bytemuck::cast_slice(entries8), storage);
        let info_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("info"), size: (n * 16) as u64, usage: storage, mapped_at_creation: false,
        });
        let agg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("agg"), size: (4 * n_wg as usize * 16 + 32) as u64, usage: storage, mapped_at_creation: false,
        });
        let mut scratch_init = vec![0u32; scratch_len];
        scratch_init[off_flags..off_flags + path_flags.len()].copy_from_slice(path_flags);
        let scratch_buf = mk_init("scratch", bytemuck::cast_slice(&scratch_init), rw);
        let sentinel = Entry { key: crate::tiler::SENTINEL_KEY, line: dummy_line, backdrop: 0, out: (n_mask_slots - 1) as u32 };
        let entries4_init = vec![sentinel; n4];
        let entries4_buf = mk_init("entries4", bytemuck::cast_slice(&entries4_init), rw);
        let mask_size = (n_mask_slots * 32 * 4) as u64;
        let partials_size = (n_records.max(1) * 64 * 4) as u64;
        let mask_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask_out"), size: mask_size, usage: rw, mapped_at_creation: false,
        });
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("partials"), size: partials_size, usage: rw, mapped_at_creation: false,
        });
        let flags_buf = mk_init("path_flags", bytemuck::cast_slice(path_flags), storage);
        let lut_buf = mk_init("mask_lut", bytemuck::cast_slice(lut), storage);

        let expand_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("expand"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lines_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: entries8_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: info_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: agg_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scratch_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: entries4_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: mask_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: partials_buf.as_entire_binding() },
            ],
        });
        let merge_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("merge"),
            layout: &p_merge.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lines_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: entries4_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mask_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: partials_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: flags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: lut_buf.as_entire_binding() },
            ],
        });

        let n_passes = 6u32;
        let n_queries = 2 * n_passes * iterations;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("timestamps"), ty: wgpu::QueryType::Timestamp, count: n_queries,
        });
        let qbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("query resolve"), size: 8 * n_queries as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let readback_size = mask_size + partials_size + (n4 * 16) as u64 + (4 * n * 4) as u64 + 16;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"), size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let qread = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("query readback"), size: 8 * n_queries as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });

        let total_iters = iterations + iterations.max(20);
        // Submit in chunks: one command buffer with hundreds of passes was lost on Metal.
        let chunk = 25u32;
        let mut it = 0u32;
        while it < total_iters {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let end = (it + chunk).min(total_iters);
            while it < end {
                // Reset per-(run, quadrant) counts and fills, and the counters.
                encoder.clear_buffer(&scratch_buf, (off_cnt * 4) as u64, Some(((off_delta - off_cnt) * 4) as u64));
                encoder.clear_buffer(&scratch_buf, (off_ctr * 4) as u64, Some(16));
                let timed = it >= total_iters - iterations;
                let base_q = if timed { 2 * n_passes * (it - (total_iters - iterations)) } else { 0 };
                let passes: [(&wgpu::ComputePipeline, &wgpu::BindGroup, u32); 6] = [
                    (&p_e1, &expand_bg, n_wg),
                    (&p_e2, &expand_bg, 1),
                    (&p_e3, &expand_bg, n_wg),
                    (&p_e5, &expand_bg, n_wg),
                    (&p_e4, &expand_bg, n_wg),
                    (&p_merge, &merge_bg, (n4 / wg_size as usize) as u32),
                ];
                for (k, (pipe, bg, groups)) in passes.iter().enumerate() {
                    let tw = timed.then(|| wgpu::ComputePassTimestampWrites {
                        query_set: &query_set,
                        beginning_of_pass_write_index: Some(base_q + 2 * k as u32),
                        end_of_pass_write_index: Some(base_q + 2 * k as u32 + 1),
                    });
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: tw });
                    pass.set_pipeline(pipe);
                    pass.set_bind_group(0, *bg, &[]);
                    pass.dispatch_workgroups(*groups, 1, 1);
                }
                it += 1;
            }
            self.queue.submit([encoder.finish()]);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut off = 0u64;
        encoder.copy_buffer_to_buffer(&mask_buf, 0, &readback, off, mask_size);
        off += mask_size;
        encoder.copy_buffer_to_buffer(&partials_buf, 0, &readback, off, partials_size);
        off += partials_size;
        encoder.copy_buffer_to_buffer(&entries4_buf, 0, &readback, off, (n4 * 16) as u64);
        off += (n4 * 16) as u64;
        encoder.copy_buffer_to_buffer(&scratch_buf, (off_rinfo * 4) as u64, &readback, off, (4 * n * 4) as u64);
        off += (4 * n * 4) as u64;
        encoder.copy_buffer_to_buffer(&scratch_buf, (off_ctr * 4) as u64, &readback, off, 16);
        encoder.resolve_query_set(&query_set, 0..n_queries, &qbuf, 0);
        encoder.copy_buffer_to_buffer(&qbuf, 0, &qread, 0, 8 * n_queries as u64);
        self.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        qread.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");

        let data = readback.slice(..).get_mapped_range().expect("mapped");
        let mut off = 0usize;
        let masks: Vec<u32> = bytemuck::cast_slice(&data[off..off + mask_size as usize]).to_vec();
        off += mask_size as usize;
        let partials: Vec<i32> = bytemuck::cast_slice(&data[off..off + partials_size as usize]).to_vec();
        off += partials_size as usize;
        let entries4: Vec<Entry> = bytemuck::cast_slice(&data[off..off + n4 * 16]).to_vec();
        off += n4 * 16;
        let run_info: Vec<u32> = bytemuck::cast_slice(&data[off..off + 4 * n * 4]).to_vec();
        off += 4 * n * 4;
        let ctr: Vec<u32> = bytemuck::cast_slice(&data[off..off + 16]).to_vec();
        drop(data);
        readback.unmap();
        let period = self.queue.get_timestamp_period() as f64;
        let qd = qread.slice(..).get_mapped_range().expect("mapped");
        let ts: &[u64] = bytemuck::cast_slice(&qd);
        let mut times_ms = Vec::new();
        for it in 0..iterations as usize {
            let mut row = [0f64; 6];
            for k in 0..6 {
                let a = ts[it * 12 + 2 * k];
                let b = ts[it * 12 + 2 * k + 1];
                row[k] = b.wrapping_sub(a) as f64 * period * 1e-6;
            }
            times_ms.push(row);
        }
        drop(qd);
        qread.unmap();
        assert!(ctr[0] as usize <= n_records, "record allocation {} exceeds host bound {}", ctr[0], n_records);
        ExpandedResult { masks, partials, entries4, run_info, n_runs: ctr[1], n_items: ctr[2], times_ms }
    }
}
