//! Opaque Ice through the real mesher: ice quads in the solid pass with the
//! opaque layer bit, and faces into ice culled. Needs prepared resources.
use super::*;
use crate::{
    assets::{Assets, RenderAssets},
    biome::Biomes,
    blocks::BlockFacts,
    chunk::Chunk,
    models,
};

struct Fixture {
    assets: Assets,
    biomes: Biomes,
    render: RenderAssets,
    visibility: horizon_world::block_render::FaceVisibility,
    tables: MeshTables,
}

fn fixture() -> Fixture {
    let root = std::env::var_os("HORIZON_MESH_TEST_RESOURCES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.local/horizon-client/26.3")
        });
    let assets = Assets::load(&root).unwrap();
    let facts = BlockFacts::load(&root, assets.registry).unwrap();
    let biomes = Biomes::load(&root).unwrap();
    let render = assets.reload(&root).unwrap();
    let visibility = models::load_visibility(&root, assets.registry).unwrap();
    let tables = MeshTables::new(&assets, &render, &facts, visibility.clone()).unwrap();
    Fixture {
        assets,
        biomes,
        render,
        visibility,
        tables,
    }
}

fn state(f: &Fixture, name: &str) -> u32 {
    f.assets.registry.default_state(name).unwrap().0
}

fn mesh(f: &Fixture, chunk: &Chunk, opaque_ice: bool) -> SectionMesh {
    let region = [[Some(chunk); 3]; 3];
    let mut mesher = Mesher {
        nether_shading: false,
        shader_pack_lighting: false,
        separate_ao: false,
        shader_pack: false,
        assets: &f.assets,
        render: &f.render,
        tables: &f.tables,
        biomes: &f.biomes,
        visibility: f.visibility.clone(),
        zoom: BiomeZoom::new(0),
        blend_radius: 0,
        smooth_lighting: false,
        cutout_leaves: true,
        translucent_layers: &[],
        leaf_culling: LeafCullingMode::Off,
        opaque_ice,
    };
    mesher.mesh(&region, 0)
}

const BASE: i32 = MIN_Y + 4;

#[test]
#[ignore = "requires prepared 26.3 resources (HORIZON_MESH_TEST_RESOURCES or .local/horizon-client/26.3)"]
fn ice_draws_in_the_solid_pass_only_when_opaque_ice_is_on() {
    let f = fixture();
    let mut chunk = Chunk::empty(0, 0);
    chunk.set(7, BASE, 7, state(&f, "minecraft:ice"));
    let vanilla = mesh(&f, &chunk, false);
    assert_eq!(vanilla.solid.len(), 0);
    assert_eq!(vanilla.translucent.len(), 24);

    let opaque = mesh(&f, &chunk, true);
    assert_eq!(opaque.translucent.len(), 0);
    assert_eq!(opaque.solid.len(), 24);
    assert!(opaque.solid.iter().all(|v| v.layer & OPAQUE_LAYER != 0));
}

#[test]
#[ignore = "requires prepared 26.3 resources (HORIZON_MESH_TEST_RESOURCES or .local/horizon-client/26.3)"]
fn opaque_ice_culls_faces_between_ice_blocks() {
    let f = fixture();
    let ice = state(&f, "minecraft:ice");
    let mut chunk = Chunk::empty(0, 0);
    chunk.set(7, BASE, 7, ice);
    chunk.set(8, BASE, 7, ice);
    let opaque = mesh(&f, &chunk, true);
    // Ten faces: the two touching ones are gone.
    assert_eq!(opaque.solid.len(), 40);
    assert_eq!(opaque.translucent.len(), 0);
}

#[test]
#[ignore = "requires prepared 26.3 resources (HORIZON_MESH_TEST_RESOURCES or .local/horizon-client/26.3)"]
fn opaque_ice_hides_faces_that_look_into_it() {
    let f = fixture();
    let stone = state(&f, "minecraft:stone");
    let ice = state(&f, "minecraft:ice");
    let mut chunk = Chunk::empty(0, 0);
    chunk.set(7, BASE, 7, stone);
    chunk.set(7, BASE + 1, 7, ice);
    let opaque = mesh(&f, &chunk, true);
    // Stone's top face and ice's bottom face are both hidden: ten quads.
    assert_eq!(opaque.solid.len(), 40);
    assert_eq!(opaque.translucent.len(), 0);
}
