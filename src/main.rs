use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};
use nalgebra::{Matrix4, Point3, Vector3};
use std::sync::Arc;

const SHADER: &str = r#"
struct Uniforms {
    mvp: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(model.position, 1.0);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#;

struct Camera {
    position: Point3<f32>,
    yaw: f32,
    pitch: f32,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl Camera {
    fn new(aspect: f32) -> Self {
        Self {
            position: Point3::new(0.0, 15.0, 0.0),
            yaw: 0.0,
            pitch: -std::f32::consts::PI / 3.0,
            fov_y: std::f32::consts::PI / 3.0,
            aspect,
            near: 0.1,
            far: 1000.0,
        }
    }

    fn forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    fn right(&self) -> Vector3<f32> {
        let forward = self.forward();
        forward.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize()
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let speed = 0.05;
        let right = self.right();
        let forward = self.forward();
        let flat_forward = Vector3::new(forward.x, 0.0, forward.z).normalize();
        self.position += right * dx * speed;
        self.position += flat_forward * dy * speed;
    }

    fn look_around(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.005;
        self.yaw += dx * sensitivity;
        self.pitch = (self.pitch - dy * sensitivity).clamp(-std::f32::consts::PI / 2.0 + 0.01, -0.01);
    }

    fn view_matrix(&self) -> Matrix4<f32> {
        let forward = self.forward();
        let right = forward.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(&forward).normalize();
        
        let target = self.position + forward;
        Matrix4::look_at_rh(&self.position, &target, &up)
    }

    fn projection_matrix(&self) -> Matrix4<f32> {
        Matrix4::new_perspective(self.aspect, self.fov_y, self.near, self.far)
    }

    fn mvp_matrix(&self) -> [f32; 16] {
        let view = self.view_matrix();
        let projection = self.projection_matrix();
        let mvp = projection * view;
        let mvp_transposed = mvp.transpose();
        let mut result = [0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                result[i * 4 + j] = mvp_transposed[(i, j)];
            }
        }
        result
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    camera: Camera,
}

impl State {
    async fn new(window: Window) -> Self {
        let size = window.inner_size();
        let window = Arc::new(window);
        
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window)).unwrap();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: wgpu::ExperimentalFeatures::default(),
                    trace: wgpu::Trace::default(),
                },
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[0f32; 16]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview_mask: None,
        });

        let vertices: [[f32; 6]; 4] = [
            [-10.0, 0.0, -10.0, 0.5, 0.5, 0.5],
            [10.0, 0.0, -10.0, 0.5, 0.5, 0.5],
            [10.0, 0.0, 10.0, 0.5, 0.5, 0.5],
            [-10.0, 0.0, 10.0, 0.5, 0.5, 0.5],
        ];

        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = indices.len() as u32;
        let aspect = config.width as f32 / config.height as f32;
        let camera = Camera::new(aspect);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            uniform_buffer,
            bind_group,
            camera,
        }
    }

    fn render(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder"),
        });

        {
            let mvp = self.camera.mvp_matrix();
            self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&mvp));

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                multiview_mask: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn handle_pan(&mut self, dx: f64, dy: f64) {
        self.camera.pan(dx as f32, dy as f32);
    }

    fn handle_look(&mut self, dx: f64, dy: f64) {
        self.camera.look_around(dx as f32, dy as f32);
    }
}

struct App {
    state: Option<State>,
    left_mouse_pressed: bool,
    right_mouse_pressed: bool,
    last_cursor_pos: Option<winit::dpi::PhysicalPosition<f64>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("3D Plane with Pan")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
        
        let window = event_loop.create_window(window_attributes).unwrap();
        
        let state = block_on(State::new(window));
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let Some(state) = &mut self.state {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::Resized(physical_size) => {
                    state.resize(physical_size);
                }
            WindowEvent::MouseInput { state: button_state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    self.left_mouse_pressed = button_state == winit::event::ElementState::Pressed;
                    if self.left_mouse_pressed {
                        self.last_cursor_pos = None;
                    }
                }
                if button == winit::event::MouseButton::Right {
                    self.right_mouse_pressed = button_state == winit::event::ElementState::Pressed;
                    if self.right_mouse_pressed {
                        self.last_cursor_pos = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.left_mouse_pressed {
                    if let Some(last_pos) = self.last_cursor_pos {
                        let dx = position.x - last_pos.x;
                        let dy = position.y - last_pos.y;
                        state.handle_look(dx, dy);
                    }
                    self.last_cursor_pos = Some(position);
                }
                if self.right_mouse_pressed {
                    if let Some(last_pos) = self.last_cursor_pos {
                        let dx = position.x - last_pos.x;
                        let dy = position.y - last_pos.y;
                        state.handle_pan(dx, dy);
                    }
                    self.last_cursor_pos = Some(position);
                }
            }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &mut self.state {
            state.render();
        }
    }
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(future)
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { 
        state: None,
        left_mouse_pressed: false,
        right_mouse_pressed: false,
        last_cursor_pos: None,
    };
    event_loop.run_app(&mut app).unwrap();
}
