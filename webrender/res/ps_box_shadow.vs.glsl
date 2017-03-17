#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

void main(void) {
    Primitive prim = load_primitive();
    //BoxShadow bs = fetch_boxshadow(prim.prim_index);
    BoxShadow bs = fetch_boxshadow(prim.prim_index);
    RectWithSize segment_rect = fetch_instance_geometry(prim.sub_index);

    VertexInfo vi = write_vertex(segment_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task);

    RenderTaskData child_task = fetch_render_task(prim.user_data.x);
    vUv.z = child_task.data1.x;

    // Constant offsets to inset from bilinear filtering border.
    vec2 patch_origin = child_task.data0.xy + vec2(1.0);
    vec2 patch_size_device_pixels = child_task.data0.zw - vec2(2.0);
    vec2 patch_size = patch_size_device_pixels / uDevicePixelRatio;

    vUv.xy = (vi.local_pos - prim.local_rect.p0) / patch_size;
    vMirrorPoint = 0.5 * prim.local_rect.size / patch_size;
    /*
    vUv.xy = (vi.local_pos - prim.local_rect.p0);
    vMirrorPoint = 0.5 * prim.local_rect.size;
    */

    vec2 texture_size = vec2(textureSize(sCache, 0));
    vCacheUvRectCoords = vec4(patch_origin, patch_origin + patch_size_device_pixels) / texture_size.xyxy;

    vOrigCoord = (vi.local_pos - prim.local_rect.p0) / prim.local_rect.size;

    vColor = bs.color;
    vBS_rect = bs.bs_rect;
    vIndex = aGlobalPrimId;
    vPatchSize = patch_size;
    vOrigin = patch_origin;
    vPatchDevice = patch_size_device_pixels;
    vTextureSize = texture_size;
    vZ = vUv.z;
    vLocalRect = vec4(prim.local_rect.p0, prim.local_rect.size);
}
