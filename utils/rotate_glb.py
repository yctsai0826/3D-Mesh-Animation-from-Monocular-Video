#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bpy
import sys
import argparse
import math

def main():
    parser = argparse.ArgumentParser(description="旋轉 GLB 模型 (保留骨架與權重)")
    parser.add_argument("--input", required=True, help="輸入的 .glb 檔案路徑")
    parser.add_argument("--output", required=True, help="輸出的 .glb 檔案路徑")
    parser.add_argument("--axis", required=True, choices=['x', 'y', 'z', 'X', 'Y', 'Z'], help="旋轉軸 (x, y, 或 z)")
    parser.add_argument("--degrees", type=float, required=True, help="旋轉角度 (度數)")

    # 處理 Blender 特有的參數傳遞方式
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    print(f"載入檔案: {args.input}")

    # 清空場景
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 匯入 GLB
    bpy.ops.import_scene.gltf(filepath=args.input)

    axis = args.axis.lower()
    radians = math.radians(args.degrees)

    # 找出所有「沒有父節點」的根物件 (Root Objects)
    # 只旋轉根物件，底下的子物件（如 Mesh）會自然跟著轉，防止發生「雙重旋轉」
    root_objects = [obj for obj in bpy.context.scene.objects if obj.parent is None]

    print(f"延 {axis.upper()} 軸旋轉 {args.degrees} 度...")
    for obj in root_objects:
        if axis == 'x':
            obj.rotation_euler[0] += radians
        elif axis == 'y':
            obj.rotation_euler[1] += radians
        elif axis == 'z':
            obj.rotation_euler[2] += radians

    # 更新場景
    bpy.context.view_layer.update()

    # 選取所有物件，將旋轉「套用 (Apply)」到底層數據中
    # 這是非常關鍵的一步！確保 Rest Pose 改變，之後動畫才不會抽搐跳回原位
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    print(f"匯出至: {args.output}")
    # 匯出 GLB (保留原有的骨架與動畫)
    bpy.ops.export_scene.gltf(
        filepath=args.output, 
        export_format='GLB',
        export_apply=True
    )
    print("完成！")

if __name__ == "__main__":
    main()