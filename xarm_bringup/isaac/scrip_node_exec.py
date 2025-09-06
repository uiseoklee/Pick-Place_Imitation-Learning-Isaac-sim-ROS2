import omni.kit.commands
from pxr import Gf, UsdGeom, Usd
import numpy as np

# A global dictionary to hold state, compatible with legacy execution mode.
NODE_STATE = {}

def setup(db: object):
    """Called once when the graph is first played or the script is updated."""
    global NODE_STATE
    NODE_STATE = {
        "prims": {
            "red_block": "/World/red_block",
            "blue_block": "/World/blue_block",
            "yellow_block": "/World/yellow_block",
            "basket": "/World/small_KLT"
        },
        "x_range": [0.3, 0.7],
        "y_range": [-0.4, 0.4],
        "z_pos": 0.07  # Spawn height
    }
    print("--- RESPAWN SCRIPT: Setup complete. Ready to respawn objects. ---")

def compute(db: object):
    """Called every time the input trigger is activated (e.g., by a key press)."""
    global NODE_STATE
    
    print("--- RESPAWN SCRIPT: Compute called. Randomizing object positions... ---")
    
    prims = NODE_STATE.get("prims", {})
    x_range = NODE_STATE.get("x_range", [0.3, 0.7])
    y_range = NODE_STATE.get("y_range", [-0.4, 0.4])
    z_pos = NODE_STATE.get("z_pos", 0.07)
    
    occupied_positions = []
    
    # 현재 스테이지 접근
    stage = omni.usd.get_context().get_stage()
    
    for name, path in prims.items():
        # 충돌 없는 새 위치 생성
        while True:
            new_pos = [
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1]),
                z_pos
            ]
            is_collision = any(np.linalg.norm(np.array(new_pos) - np.array(p)) < 0.15 for p in occupied_positions)
            if not is_collision:
                occupied_positions.append(new_pos)
                break
        
        try:
            # 방법 1: 직접 USD API 사용 (가장 신뢰할 수 있는 방법)
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim)
                # 기존 변환 초기화
                xform.ClearXformOpOrder()
                # 새 변환 설정
                xform_op = xform.AddTranslateOp()
                xform_op.Set(Gf.Vec3d(new_pos[0], new_pos[1], new_pos[2]))
                print(f"Successfully moved {name} to: {new_pos} (USD API method)")
                continue
        except Exception as e:
            print(f"[INFO] USD API method failed for {name}: {e}")
        
        try:
            # 방법 2: TransformPrim 명령 사용 (path 매개변수)
            omni.kit.commands.execute('TransformPrim',
                path=path,
                translation=Gf.Vec3d(new_pos[0], new_pos[1], new_pos[2])
            )
            print(f"Successfully moved {name} to: {new_pos} (path parameter)")
        except Exception as e:
            try:
                # 방법 3: TransformPrim 명령 사용 (prim_path 매개변수)
                omni.kit.commands.execute('TransformPrim',
                    prim_path=path,
                    translation=Gf.Vec3d(new_pos[0], new_pos[1], new_pos[2])
                )
                print(f"Successfully moved {name} to: {new_pos} (prim_path parameter)")
            except Exception as e2:
                print(f"[ERROR] Failed to move {name}. Errors: {e}, {e2}")

    return True

def cleanup(db: object):
    """Called when the graph is stopped or the node is deleted."""
    global NODE_STATE
    NODE_STATE = {}
    print("--- RESPAWN SCRIPT: Cleanup complete. ---")
