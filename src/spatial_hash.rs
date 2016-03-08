/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::{Rect, Size2D, Point2D, Matrix4, Point4D};
use internal_types::{PackedTile, RenderTile};
use std::{cmp, mem};
use webrender_traits::{ColorF};
//use internal_types::{PackedSceneItem, PackedSceneVertex};
use internal_types::{PackedCircle};

struct CirclePrimitive {
    center: Point2D<f32>,
    outer_radius: f32,
    inner_radius: f32,
    color: ColorF,
}

struct RectanglePrimitive {
    positions: [Point4D<f32>; 4],
    color: ColorF,
    outer_radius: Size2D<f32>,
    inner_radius: Size2D<f32>,
    ref_point: Point2D<f32>,
}

enum Primitive {
    Rectangle(RectanglePrimitive),
    Circle(CirclePrimitive),
}

#[derive(Copy, Clone, Debug)]
struct ItemKey(u32);

struct Bucket {
    position: Point2D<i32>,        // For debugging / validation purposes only
    items: Vec<ItemKey>,
}

impl Bucket {
    fn new(position: Point2D<i32>) -> Bucket {
        Bucket {
            position: position,
            items: Vec::new(),
        }
    }
}

pub struct SpatialHash {
    primitives: Vec<Primitive>,
    buckets: Vec<Bucket>,
    scene_size: Size2D<i32>,
    pub tile_size: Size2D<i32>,
    x_tile_count: i32,
    y_tile_count: i32,
}

#[derive(Debug)]
struct TransformedRect {
    vertices: [Point4D<f32>; 4],
    screen_min: Point2D<i32>,
    screen_max: Point2D<i32>,
}

#[derive(Debug, Clone)]
pub struct DebugTile {
    pub position: Point2D<i32>,
    pub item_count: usize,
}

#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub tile_size: Size2D<i32>,
    pub tiles: Vec<DebugTile>,
}

impl DebugInfo {
    pub fn empty() -> DebugInfo {
        DebugInfo {
            tile_size: Size2D::zero(),
            tiles: Vec::new(),
        }
    }
}

fn transform_rect(rect: &Rect<f32>,
                  transform: &Matrix4) -> TransformedRect {
    let vertices = [
        transform.transform_point4d(&Point4D::new(rect.origin.x,
                                                  rect.origin.y,
                                                  0.0,
                                                  1.0)),
        transform.transform_point4d(&Point4D::new(rect.bottom_left().x,
                                                  rect.bottom_left().y,
                                                  0.0,
                                                  1.0)),
        transform.transform_point4d(&Point4D::new(rect.bottom_right().x,
                                                  rect.bottom_right().y,
                                                  0.0,
                                                  1.0)),
        transform.transform_point4d(&Point4D::new(rect.top_right().x,
                                                  rect.top_right().y,
                                                  0.0,
                                                  1.0)),
    ];

    let mut screen_min = Point2D::new( 1000000,  1000000);
    let mut screen_max = Point2D::new(-1000000, -1000000);

    for vertex in &vertices {
        let inv_w = 1.0 / vertex.w;
        let vx = vertex.x * inv_w;
        let vy = vertex.y * inv_w;
        screen_min.x = cmp::min(screen_min.x, vx.floor() as i32);
        screen_min.y = cmp::min(screen_min.y, vy.floor() as i32);
        screen_max.x = cmp::max(screen_max.x, vx.ceil() as i32);
        screen_max.y = cmp::max(screen_max.y, vy.ceil() as i32);
    }

    TransformedRect {
        vertices: vertices,
        screen_min: screen_min,
        screen_max: screen_max,
    }
}

impl SpatialHash {
    pub fn new(scene_size: Size2D<i32>,
               tile_size: Size2D<i32>) -> SpatialHash {
        // TODO(gw) Can relax this later, but simplifies things for now.
        debug_assert!(scene_size.width % tile_size.width == 0);
        debug_assert!(scene_size.height % tile_size.height == 0);

        let x_tile_count = scene_size.width / tile_size.width;
        let y_tile_count = scene_size.height / tile_size.height;

        let mut buckets = Vec::with_capacity((x_tile_count * y_tile_count) as usize);
        for y in 0..y_tile_count {
            for x in 0..x_tile_count {
                let position = Point2D::new(x * tile_size.width, y * tile_size.height);
                let bucket = Bucket::new(position);
                buckets.push(bucket);
            }
        }

        SpatialHash {
            scene_size: scene_size,
            tile_size: tile_size,
            x_tile_count: x_tile_count,
            y_tile_count: y_tile_count,
            primitives: Vec::new(),
            buckets: buckets,
        }
    }

    pub fn add_circle(&mut self,
                      center: &Point2D<f32>,
                      color: &ColorF,
                      outer_radius: f32,
                      inner_radius: f32) {
        let primitive = Primitive::Circle(CirclePrimitive {
            center: *center,
            outer_radius: outer_radius,
            inner_radius: inner_radius,
            color: *color,
        });

        let item_key = ItemKey(self.primitives.len() as u32);

        let screen_min_x = (center.x - outer_radius).floor() as i32;
        let screen_min_y = (center.y - outer_radius).floor() as i32;
        let screen_max_x = (center.x + outer_radius).ceil() as i32;
        let screen_max_y = (center.y + outer_radius).ceil() as i32;

        let tile_x0 = screen_min_x / self.tile_size.width;
        let tile_y0 = screen_min_y / self.tile_size.height;
        let tile_x1 = (screen_max_x + self.tile_size.width - 1) / self.tile_size.width;
        let tile_y1 = (screen_max_y + self.tile_size.height - 1) / self.tile_size.height;

        let tile_x0 = cmp::min(tile_x0, self.x_tile_count);
        let tile_x0 = cmp::max(tile_x0, 0);
        let tile_x1 = cmp::min(tile_x1, self.x_tile_count);
        let tile_x1 = cmp::max(tile_x1, 0);

        let tile_y0 = cmp::min(tile_y0, self.y_tile_count);
        let tile_y0 = cmp::max(tile_y0, 0);
        let tile_y1 = cmp::min(tile_y1, self.y_tile_count);
        let tile_y1 = cmp::max(tile_y1, 0);

        self.primitives.push(primitive);

        for y in tile_y0..tile_y1 {
            for x in tile_x0..tile_x1 {
                let bucket = &mut self.buckets[(y * self.x_tile_count + x) as usize];
                bucket.items.push(item_key);
            }
        }
    }

    pub fn add_color_rectangle(&mut self,
                               rect: &Rect<f32>,
                               color: &ColorF,
                               transform: &Matrix4,
                               outer_border_radius: &Size2D<f32>,
                               inner_border_radius: &Size2D<f32>,
                               ref_point: &Point2D<f32>) {
        /*
        let xf_rect = transform_rect(rect, transform);

        let primitive = Primitive::Rectangle(RectanglePrimitive {
            positions: xf_rect.vertices,
            color: *color,
            outer_radius: *outer_border_radius,
            inner_radius: *inner_border_radius,
            ref_point: *ref_point
        });

        let item_key = ItemKey(self.primitives.len() as u32);

        let tile_x0 = xf_rect.screen_min.x / self.tile_size.width;
        let tile_y0 = xf_rect.screen_min.y / self.tile_size.height;
        let tile_x1 = (xf_rect.screen_max.x + self.tile_size.width - 1) / self.tile_size.width;
        let tile_y1 = (xf_rect.screen_max.y + self.tile_size.height - 1) / self.tile_size.height;

        let tile_x0 = cmp::min(tile_x0, self.x_tile_count);
        let tile_x0 = cmp::max(tile_x0, 0);
        let tile_x1 = cmp::min(tile_x1, self.x_tile_count);
        let tile_x1 = cmp::max(tile_x1, 0);

        let tile_y0 = cmp::min(tile_y0, self.y_tile_count);
        let tile_y0 = cmp::max(tile_y0, 0);
        let tile_y1 = cmp::min(tile_y1, self.y_tile_count);
        let tile_y1 = cmp::max(tile_y1, 0);

        self.primitives.push(primitive);

        //println!("add_color_rectangle {:?} {:?} {},{} -> {},{}", xf_rect, self.tile_size, tile_x0, tile_y0, tile_x1, tile_y1);

        for y in tile_y0..tile_y1 {
            for x in tile_x0..tile_x1 {
                //println!("\t{} {}", x, y);
                let bucket = &mut self.buckets[(y * self.x_tile_count + x) as usize];

                if xf_rect.screen_min.x < bucket.position.x &&
                   xf_rect.screen_min.y < bucket.position.y &&
                   xf_rect.screen_max.x > bucket.position.x + self.tile_size.width &&
                   xf_rect.screen_max.y > bucket.position.y + self.tile_size.height &&
                   color.a == 1.0 {
                    if bucket.items.len() > 0 {
                        println!("{:?} clears {} items!", item_key, bucket.items.len());
                    }
                    bucket.items.clear();
                }

                bucket.items.push(item_key);
            }
        }*/
    }

    pub fn build(&self) -> Vec<RenderTile> {
        let mut tiles = Vec::new();

        for bucket in &self.buckets {
            let mut packed_tiles = Vec::new();

            if bucket.items.len() > 0 {
                let mut circles: [PackedCircle; 256] = unsafe { mem::zeroed() };

                for (index, item_key) in bucket.items.iter().enumerate() {
                    let prim = &self.primitives[item_key.0 as usize];

                    match prim {
                        &Primitive::Rectangle(ref rect) => {
                            panic!("todo");
                        }
                        &Primitive::Circle(ref circle) => {
                            let packed_circle = PackedCircle {
                                color: [
                                    circle.color.r,
                                    circle.color.g,
                                    circle.color.b,
                                    circle.color.a
                                ],
                                center_outer_inner_radius: [
                                    circle.center.x,
                                    circle.center.y,
                                    circle.outer_radius,
                                    circle.inner_radius,
                                ],
                            };
                            circles[index] = packed_circle;
                        }
                    }
                }

                packed_tiles.push(PackedTile {
                    circles: circles,
                    circle_count: [bucket.items.len() as f32, 0.0, 0.0, 0.0]
                });

                /*
                let mut items: [PackedSceneItem; 256] = unsafe { mem::zeroed() };

                for (index, item_key) in bucket.items.iter().enumerate() {
                    let prim = &self.primitives[item_key.0 as usize];

                    match prim {
                        &Primitive::Rectangle(ref rect) => {
                            let packed_item = PackedSceneItem {
                                vertices: [
                                    PackedSceneVertex::new(&rect.positions[0], &rect.color, &Point2D::zero()),
                                    PackedSceneVertex::new(&rect.positions[1], &rect.color, &Point2D::zero()),
                                    PackedSceneVertex::new(&rect.positions[2], &rect.color, &Point2D::zero()),
                                    PackedSceneVertex::new(&rect.positions[3], &rect.color, &Point2D::zero()),
                                ],
                                radius: [
                                    rect.outer_radius.width,
                                    rect.outer_radius.height,
                                    rect.inner_radius.width,
                                    rect.inner_radius.height,
                                ],
                                ref_point: [
                                    rect.ref_point.x,
                                    rect.ref_point.y,
                                    0.0,
                                    0.0
                                ]
                            };
                            items[index] = packed_item;
                        }
                    }
                }

                packed_tiles.push(PackedTile {
                    items: items,
                    rect_count: [bucket.items.len() as f32, 0.0, 0.0, 0.0]
                });
                */
            }

            tiles.push(RenderTile {
                origin: bucket.position,
                packed_tiles: packed_tiles,
            });
        }

        tiles
    }

    pub fn build_debug_info(&self) -> DebugInfo {
        let mut info = DebugInfo {
            tile_size: self.tile_size,
            tiles: Vec::new(),
        };

        for bucket in &self.buckets {
            info.tiles.push(DebugTile {
                position: bucket.position,
                item_count: bucket.items.len(),
            });
        }

        info
    }
}
