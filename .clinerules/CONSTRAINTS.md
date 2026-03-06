# 🔥 CRITICAL FILE SYSTEM CONSTRAINTS (MUST FOLLOW) 🔥

## 🚫 FORBIDDEN ACTIONS (禁止事项)
1. **NO NEW ROOT DIRECTORIES**: You are STRICTLY PROHIBITED from creating new folders in the project root (e.g., `output_optimized`, `output_iter1`, `test_results`).
2. **NO ROOT SCRIPTS**: Do not create temporary python scripts (`test.py`, `debug.py`) in the root.
3. **NO VISUAL POPUPS**: Do not use `plt.show()` or `cv2.imshow()`.

## ✅ MANDATORY PATH RULES (强制路径规则)
1. **TEMP SCRIPTS**: All temporary analysis/test scripts MUST be created in `temp/`.
   - *Fixing Imports*: When running scripts from `temp/`, you MUST append the project root to `sys.path` to import `automakeosufile`.
   - Example snippet for EVERY temp script:
     ```python
     import sys, os
     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
     from automakeosufile import ...
     ```
2. **OUTPUT LOCATION**: 
   - Standard output -> `output/`
   - Optimization iterations -> `output/optimization_experiments/` (Create this subfolder if needed)
   - NEVER create folders like `output_optimized_v2` in root.

## 🧹 CLEANUP PROTOCOL
- If you create a file by mistake in the root, DELETE IT immediately.