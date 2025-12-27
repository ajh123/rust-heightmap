# Rust Heightmap Renderer

This project is a simple heightmap renderer written in Rust. It renders a basic flat plane split into chunks, using `wgpu` for graphics rendering and `winit` for window management.

## Features

- Renders a flat plane divided into chunks.
- Utilizes `wgpu` for efficient GPU rendering.
- Window management with `winit`.
- Basic camera controls for navigating the scene.
- Chunk-based terrain generation with support for flat and Perlin noise-based terrains.
- Modular design for easy extension and modification.

## Next Steps

Future improvements could include:
- Implementing heightmap loading from image files.
- Adding lighting and shading effects.
- Adding texture mapping to the heightmap.
- Incorporating GUI elements for better user interaction.

### Development timeline

## First Pass:

The first pass was implemented in two stages:
1. **Basic Setup**: Created a window using `winit` and set up a rendering loop with `wgpu`. This stage involved initializing the GPU, creating a swap chain, and rendering a simple 3D plane. *(commit `3d58d74c70df3ca1c6eb8f75c59eeb0556a16a4c`)*
2. **Chunking**: Divided the plane into smaller chunks to optimize rendering performance. Each chunk is rendered separately, allowing for better management of large heightmaps. *(commit `b286658a382ea06016a26bf3d3420fa559ce5702`)*

This first pass was implemented with the GLM-4.7 assistant on OpenCode in roughly one hour, *(see conversation: https://opncd.ai/share/HhULZY4C)*

![](./docs/Screenshot%202025-12-27%20123805.png)
*Screenshot of the heightmap renderer in action.*

### Second Pass:

The second pass focused on enhancing the terrain generation capabilities:
1. **Terrain Generation**: Implemented a basic terrain generator that creates flat terrain chunks. This involved defining a `TerrainGenerator` trait and a `FlatTerrainGenerator` struct that generates height data for each chunk. *(commit `7b62d9769d9b6b0e52c9bb8c2491b55b3e957471`)*
2. **Noise-based Terrain**: Added a `PerlinTerrainGenerator` that generates more complex terrain using Perlin noise. This allows for more realistic heightmaps and varied landscapes. *(commit `b15b506a3a0d2dc02e1fc958d3a711ae08b5d7d0`)*

This second pass was mostly implemented by hand, with some assistance from the GLM-4.7 model on OpenCode for chunk grid simplification. 

![](./docs/Screenshot%202025-12-27%20132051.png)
*Screenshot showcasing the Perlin noise-generated terrain.*

## License

This project is dual-licensed under the MIT License and the Apache License 2.0. See the [`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE) files for details.
Feel free to use and modify the code as per the terms of these licenses.
