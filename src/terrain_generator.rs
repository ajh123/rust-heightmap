use parrot::Perlin;

pub trait TerrainGenerator {
    fn generate_chunk(&self, chunk_x: i32, chunk_z: i32) -> Vec<f32>;
}

pub struct FlatTerrainGenerator {
    chunk_size: usize,
}

impl TerrainGenerator for FlatTerrainGenerator {
    fn generate_chunk(&self, _chunk_x: i32, _chunk_z: i32) -> Vec<f32> {
        let vertex_count = (self.chunk_size + 1) * (self.chunk_size + 1);
        vec![0.0; vertex_count]
    }
}

impl FlatTerrainGenerator {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }
}

pub struct PerlinTerrainGenerator {
    perlin: Perlin,
    chunk_size: usize,
    scale: f64,
    amplitude: f32,
}

impl PerlinTerrainGenerator {
    pub fn new(chunk_size: usize, seed: u64, scale: f64, amplitude: f32) -> Self {
        Self {
            perlin: Perlin::new(seed),
            chunk_size,
            scale,
            amplitude,
        }
    }
}

impl TerrainGenerator for PerlinTerrainGenerator {
    fn generate_chunk(&self, chunk_x: i32, chunk_z: i32) -> Vec<f32> {
        let mut heights = Vec::with_capacity((self.chunk_size + 1) * (self.chunk_size + 1));

        for iz in 0..=self.chunk_size {
            for ix in 0..=self.chunk_size {
                let world_x = (chunk_x * self.chunk_size as i32 + ix as i32) as f64;
                let world_z = (chunk_z * self.chunk_size as i32 + iz as i32) as f64;

                let x = world_x * self.scale;
                let y = world_z * self.scale;
                let height = self.perlin.noise2d(x, y);
                heights.push(height as f32 * self.amplitude);
            }
        }

        heights
    }
}