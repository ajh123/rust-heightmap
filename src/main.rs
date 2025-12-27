mod terrain_generator;
mod world;
mod world_renderer;

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
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    camera_position: vec4<f32>,
    light_params: vec2<f32>,
}

struct PointLight {
    position: vec4<f32>,
    color_intensity: vec4<f32>,
    radius: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage, read> point_lights: array<PointLight>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) metallic: f32,
    @location(4) roughness: f32,
    @location(5) ao: f32,
    @location(6) subsurface: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) metallic: f32,
    @location(4) roughness: f32,
    @location(5) ao: f32,
    @location(6) subsurface: f32,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvp * vec4<f32>(model.position, 1.0);
    out.world_position = model.position;
    out.color = model.color;
    out.normal = normalize(model.normal);
    out.metallic = model.metallic;
    out.roughness = model.roughness;
    out.ao = model.ao;
    out.subsurface = model.subsurface;
    return out;
}

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH = max(dot(N, H), 0.0);
    let NdotH2 = NdotH * NdotH;

    let num = a2;
    var denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265 * denom * denom;

    return num / denom;
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;

    let num = NdotV;
    let denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx2 = geometry_schlick_ggx(NdotV, roughness);
    let ggx1 = geometry_schlick_ggx(NdotL, roughness);

    return ggx1 * ggx2;
}

fn fresnel_schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn calculate_point_light(world_pos: vec3<f32>, light: PointLight,
                        N: vec3<f32>, V: vec3<f32>, F0: vec3<f32>, albedo: vec3<f32>,
                        metallic: f32, roughness: f32, sss: f32) -> vec3<f32> {
    let L_dir = light.position.xyz - world_pos;
    let distance = length(L_dir);

    if (distance > light.radius) {
        return vec3<f32>(0.0);
    }

    let L_norm = normalize(L_dir);
    let H_norm = normalize(V + L_norm);

    let attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
    var volumetric = 1.0 - (distance / light.radius);
    volumetric = pow(volumetric, 2.0);

    let L_color = light.color_intensity.xyz;
    let L_intensity = light.color_intensity.w;

    let radiance = L_color * L_intensity * attenuation;

    let F = fresnel_schlick(max(dot(H_norm, V), 0.0), F0);
    let NDF = distribution_ggx(N, H_norm, roughness);
    let G = geometry_smith(N, V, L_norm, roughness);

    let numerator = NDF * G * F;
    let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L_norm), 0.0) + 0.0001;
    let specular = numerator / denominator;

    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD *= 1.0 - metallic;

    let NdotL = max(dot(N, L_norm), 0.0);

    var indirect = vec3<f32>(0.0);
    if (sss > 0.01 && NdotL < 0.0) {
        indirect = L_color * L_intensity * attenuation * sss * 0.3 * abs(NdotL);
    }

    let direct = (kD * albedo / 3.14159265 + specular) * radiance * NdotL * volumetric;

    return direct + indirect;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let N = normalize(in.normal);
    let V = normalize(uniforms.camera_position.xyz - in.world_position);
    
    let sun_dir = normalize(uniforms.light_direction.xyz);
    let H_sun = normalize(V + sun_dir);
    
    let F0 = mix(vec3<f32>(0.04), in.color, in.metallic);
    
    var Lo = vec3<f32>(0.0);
    
    let radiance_sun = uniforms.light_color.xyz * 1.0;
    let F_sun = fresnel_schlick(max(dot(H_sun, V), 0.0), F0);
    let NDF_sun = distribution_ggx(N, H_sun, in.roughness);
    let G_sun = geometry_smith(N, V, sun_dir, in.roughness);
    
    let numerator_sun = NDF_sun * G_sun * F_sun;
    let denominator_sun = 4.0 * max(dot(N, V), 0.0) * max(dot(N, sun_dir), 0.0) + 0.0001;
    let specular_sun = numerator_sun / denominator_sun;
    
    let kS_sun = F_sun;
    var kD_sun = vec3<f32>(1.0) - kS_sun;
    kD_sun *= 1.0 - in.metallic;
    
    let NdotL_sun = max(dot(N, sun_dir), 0.0);
    let direct_sun = (kD_sun * in.color / 3.14159265 + specular_sun) * radiance_sun * NdotL_sun;
    
    var indirect_sun = vec3<f32>(0.0);
    if (in.subsurface > 0.01 && NdotL_sun < 0.0) {
        indirect_sun = uniforms.light_color.xyz * 0.2 * in.subsurface * abs(NdotL_sun);
    }
    
    Lo = direct_sun + indirect_sun;
    
    for (var i: u32 = 0u; i < u32(uniforms.light_params.y); i = i + 1u) {
        let light = point_lights[i];
        Lo += calculate_point_light(
            in.world_position, light,
            N, V, F0, in.color, in.metallic, in.roughness, in.subsurface
        );
    }
    
    let ambient = vec3<f32>(uniforms.light_params.x * 0.05) * in.color * in.ao;
    let color = ambient + Lo;
    
    return vec4<f32>(color, 1.0);
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

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct LightUniforms {
    mvp: [f32; 16],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    camera_position: [f32; 4],
    light_params: [f32; 2],
    _padding: [f32; 2],
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    camera: Camera,
    world: world::World,
    world_renderer: world_renderer::WorldRenderer,
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

        let light_uniforms = LightUniforms {
            mvp: [0f32; 16],
            light_direction: [0.5, 0.8, 0.3, 0.0],
            light_color: [1.0, 0.95, 0.9, 0.0],
            camera_position: [0.0, 15.0, 0.0, 0.0],
            light_params: [0.3, 0.0],
            _padding: [0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[light_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let world_renderer = world_renderer::WorldRenderer::new(&device, &uniform_buffer);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[world_renderer.bind_group_layout()],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 13]>() as wgpu::BufferAddress,
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
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 3) as wgpu::BufferAddress,
                            shader_location: 3,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 3 + std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                            shader_location: 4,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 3 + std::mem::size_of::<f32>() * 2) as wgpu::BufferAddress,
                            shader_location: 5,
                            format: wgpu::VertexFormat::Float32,
                        },
                        wgpu::VertexAttribute {
                            offset: (std::mem::size_of::<[f32; 3]>() * 3 + std::mem::size_of::<f32>() * 3) as wgpu::BufferAddress,
                            shader_location: 6,
                            format: wgpu::VertexFormat::Float32,
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

        let aspect = config.width as f32 / config.height as f32;
        let camera = Camera::new(aspect);

        let terrain_generator = Box::new(terrain_generator::PerlinTerrainGenerator::new(
            world::CHUNK_SIZE,
            42,
            0.1,
            10.0,
        ));
        let mut world = world::World::new(terrain_generator);
        world.update(camera.position.x, camera.position.z, 4);

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            camera,
            world,
            world_renderer,
        }
    }

    fn render(&mut self) {
        self.world.update(self.camera.position.x, self.camera.position.z, 4);
        self.world_renderer.update(&self.device, &self.queue, &mut self.world);

        let output = self.surface.get_current_texture().unwrap();
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Encoder"),
        });

        {
            let mvp = self.camera.mvp_matrix();
            let camera_pos = [self.camera.position.x, self.camera.position.y, self.camera.position.z];
            let num_lights = self.world.lights.len() as u32;
            
            let light_uniforms = LightUniforms {
                mvp,
                light_direction: [0.5, 0.8, 0.3, 0.0],
                light_color: [1.0, 0.95, 0.9, 0.0],
                camera_position: [camera_pos[0], camera_pos[1], camera_pos[2], 0.0],
                light_params: [0.3, num_lights as f32],
                _padding: [0.0, 0.0],
            };
            self.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[light_uniforms]));
            self.world_renderer.update_lights(&self.queue, &self.world.lights);

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
            self.world_renderer.render(&mut render_pass);
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
            .with_title("Rust Heightmap Viewer")
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
