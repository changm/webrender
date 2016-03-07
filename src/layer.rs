/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use spring::{DAMPING, STIFFNESS, Spring};
use euclid::{Point2D, Rect, Size2D, Matrix4};

/*
use aabbtree::{AABBTree, NodeIndex};
use internal_types::{BatchUpdate, BatchUpdateList, BatchUpdateOp, DisplayItemId};
use internal_types::{DrawListItemIndex, DrawListId, DrawListGroupId};
use webrender_traits::ScrollLayerId;

pub struct Layer {
    // TODO: Remove pub from here if possible in the future
    pub aabb_tree: AABBTree,
    pub scrolling: ScrollingState,
    pub viewport_size: Size2D<f32>,
    pub layer_size: Size2D<f32>,
    pub world_origin: Point2D<f32>,
    pub local_transform: Matrix4,
    pub world_transform: Matrix4,
    pub children: Vec<ScrollLayerId>,
}

impl Layer {
    pub fn new(world_origin: Point2D<f32>,
               layer_size: Size2D<f32>,
               viewport_size: Size2D<f32>,
               transform: Matrix4) -> Layer {
        let rect = Rect::new(Point2D::zero(), layer_size);
        let aabb_tree = AABBTree::new(1024.0, &rect);

        Layer {
            aabb_tree: aabb_tree,
            scrolling: ScrollingState::new(),
            viewport_size: viewport_size,
            world_origin: world_origin,
            layer_size: layer_size,
            local_transform: transform,
            world_transform: transform,
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child: ScrollLayerId) {
        self.children.push(child);
    }

    pub fn reset(&mut self, pending_updates: &mut BatchUpdateList) {
        for node in &mut self.aabb_tree.nodes {
            /*
            if let Some(ref mut compiled_node) = node.compiled_node {
                let vertex_buffer_id = compiled_node.vertex_buffer_id.take().unwrap();
                pending_updates.push(BatchUpdate {
                    id: vertex_buffer_id,
                    op: BatchUpdateOp::Destroy,
                });
            }*/
        }
    }

    #[inline]
    pub fn insert(&mut self,
                  rect: Rect<f32>,
                  draw_list_group_id: DrawListGroupId,
                  draw_list_id: DrawListId,
                  item_index: DrawListItemIndex) {
        self.aabb_tree.insert(rect,
                              draw_list_group_id,
                              draw_list_id,
                              item_index);
    }

    pub fn finalize(&mut self, scrolling: &ScrollingState) {
        self.scrolling = *scrolling;
        self.aabb_tree.finalize();
    }

    pub fn cull(&mut self) {
        // TODO(gw): Take viewport_size into account here!!!
        let viewport_rect = Rect::new(Point2D::zero(), self.viewport_size);
        let adjusted_viewport = viewport_rect.translate(&-self.scrolling.offset);
        self.aabb_tree.cull(&adjusted_viewport);
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        self.aabb_tree.print(NodeIndex(0), 0);
    }

    pub fn overscroll_amount(&self) -> Size2D<f32> {
        let overscroll_x = if self.scrolling.offset.x > 0.0 {
            -self.scrolling.offset.x
        } else if self.scrolling.offset.x < self.viewport_size.width - self.layer_size.width {
            self.viewport_size.width - self.layer_size.width - self.scrolling.offset.x
        } else {
            0.0
        };

        let overscroll_y = if self.scrolling.offset.y > 0.0 {
            -self.scrolling.offset.y
        } else if self.scrolling.offset.y < self.viewport_size.height - self.layer_size.height {
            self.viewport_size.height - self.layer_size.height - self.scrolling.offset.y
        } else {
            0.0
        };

        Size2D::new(overscroll_x, overscroll_y)
    }

    pub fn stretch_overscroll_spring(&mut self) {
        let overscroll_amount = self.overscroll_amount();
        self.scrolling.spring.coords(self.scrolling.offset,
                                     self.scrolling.offset,
                                     self.scrolling.offset + overscroll_amount);
    }

    pub fn tick_scrolling_bounce_animation(&mut self) {
        self.scrolling.spring.animate();
        self.scrolling.offset = self.scrolling.spring.current();
    }
}
*/

#[derive(Copy, Clone)]
pub struct ScrollingState {
    pub offset: Point2D<f32>,
    pub spring: Spring,
    pub started_bouncing_back: bool,
}

impl ScrollingState {
    pub fn new() -> ScrollingState {
        ScrollingState {
            offset: Point2D::new(0.0, 0.0),
            spring: Spring::at(Point2D::new(0.0, 0.0), STIFFNESS, DAMPING),
            started_bouncing_back: false,
        }
    }
}
