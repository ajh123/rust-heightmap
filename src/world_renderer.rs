use crate::world::{World, Chunk, ChunkKey, LightSource, CHUNK_SIZE};
use wgpu::util::DeviceExt;
use std::collections::HashMap;

#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
struct PointLightGPU {
    position: [f32; 4],
    color_intensity: [f32; 4],
    radius: f32,
    _pad: f32,
}

struct ChunkBuffers {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

pub struct WorldRenderer {
    buffers: HashMap<ChunkKey, ChunkBuffers>,
    index_buffer: Option<wgpu::Buffer>,
    index_count: u32,
    point_light_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

fn calculate_normal(heights: &[f32], ix: usize, iz: usize) -> [f32; 3] {
    let size = CHUNK_SIZE + 1;
    
    let h_center = heights[iz * size + ix];
    
    let h_left = if ix > 0 { heights[iz * size + (ix - 1)] } else { h_center };
    let h_right = if ix < CHUNK_SIZE { heights[iz * size + (ix + 1)] } else { h_center };
    let h_up = if iz > 0 { heights[(iz - 1) * size + ix] } else { h_center };
    let h_down = if iz < CHUNK_SIZE { heights[(iz + 1) * size + ix] } else { h_center };
    
    let normal = nalgebra::Vector3::new(
        (h_left - h_right) / 2.0,
        1.0,
        (h_down - h_up) / 2.0,
    ).normalize();
    
    [normal.x, normal.y, normal.z]
}

impl WorldRenderer {
    pub fn new(device: &wgpu::Device, uniform_buffer: &wgpu::Buffer) -> Self {
        let max_lights = 64;
        let point_lights: Vec<PointLightGPU> = vec![PointLightGPU {
            position: [0.0, 0.0, 0.0, 0.0],
            color_intensity: [0.0, 0.0, 0.0, 0.0],
            radius: 0.0,
            _pad: 0.0,
        }; max_lights];

        let point_light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Light Buffer"),
            contents: bytemuck::cast_slice(&point_lights),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("World Renderer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World Renderer Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: point_light_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            buffers: HashMap::new(),
            index_buffer: None,
            index_count: 0,
            point_light_buffer,
            bind_group: Some(bind_group),
            bind_group_layout: Some(bind_group_layout),
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, world: &mut World) {
        if self.index_buffer.is_none() {
            let indices = Chunk::generate_indices();
            self.index_count = indices.len() as u32;
            self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("World Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            }));
        }

        for (key, chunk) in world.chunks_iter_mut() {
            if chunk.dirty {
                let mut vertices = Vec::with_capacity(Chunk::vertex_count() * 6);
                let world_x = key.x as f32 * CHUNK_SIZE as f32;
                let world_z = key.z as f32 * CHUNK_SIZE as f32;

                for (vertex_index, &height) in chunk.heights.iter().enumerate() {
                    let (ix, iz) = Chunk::get_grid_position(vertex_index);
                    let x = world_x + ix as f32;
                    let z = world_z + iz as f32;

                    let base_color = if height > 6.0 {
                        [0.55, 0.50, 0.45]
                    } else if height > 3.0 {
                        [0.60, 0.65, 0.50]
                    } else if height < 1.0 {
                        [0.45, 0.50, 0.60]
                    } else {
                        [0.52, 0.68, 0.45]
                    };

                    let normal = calculate_normal(&chunk.heights, ix, iz);

                    let metallic = chunk.metallic[vertex_index];
                    let roughness = chunk.roughness[vertex_index];
                    let ao = chunk.ao[vertex_index];
                    let subsurface = chunk.subsurface[vertex_index];

                    vertices.extend_from_slice(&[x, height, z]);
                    vertices.extend_from_slice(&base_color);
                    vertices.extend_from_slice(&normal);
                    vertices.push(metallic);
                    vertices.push(roughness);
                    vertices.push(ao);
                    vertices.push(subsurface);
                }

                if let Some(buffers) = self.buffers.get_mut(key) {
                    queue.write_buffer(&buffers.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
                } else {
                    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Chunk Vertex Buffer"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });

                    self.buffers.insert(*key, ChunkBuffers {
                        vertex_buffer,
                        index_buffer: self.index_buffer.as_ref().unwrap().clone(),
                        num_indices: self.index_count,
                    });
                }

                chunk.dirty = false;
            }
        }

        let active_keys: std::collections::HashSet<_> = world.chunks_iter().map(|(k, _)| *k).collect();
        self.buffers.retain(|key, _| active_keys.contains(key));
    }

    pub fn update_lights(&mut self, queue: &wgpu::Queue, lights: &[LightSource]) {
        let max_lights = 64;
        let point_lights_gpu: Vec<PointLightGPU> = lights.iter().map(|l| PointLightGPU {
            position: [l.position[0], l.position[1], l.position[2], 0.0],
            color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
            radius: l.radius,
            _pad: 0.0,
        }).collect();

        let mut padded_lights = vec![PointLightGPU {
            position: [0.0, 0.0, 0.0, 0.0],
            color_intensity: [0.0, 0.0, 0.0, 0.0],
            radius: 0.0,
            _pad: 0.0,
        }; max_lights];

        for (i, light) in point_lights_gpu.iter().enumerate() {
            if i < max_lights {
                padded_lights[i] = *light;
            }
        }

        queue.write_buffer(&self.point_light_buffer, 0, bytemuck::cast_slice(&padded_lights));
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group.as_ref().unwrap()
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.bind_group_layout.as_ref().unwrap()
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_bind_group(0, self.bind_group(), &[]);
        for (_, buffers) in &self.buffers {
            render_pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
            render_pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..buffers.num_indices, 0, 0..1);
        }
    }
}

impl Default for WorldRenderer {
    fn default() -> Self {
        panic!("WorldRenderer::default() not supported, use WorldRenderer::new() with device and uniform_buffer")
    }
}
