| 参数 (Parameter) | 符号 (Symbol) | 值 (Value) | 单位 (Unit) |
| :--- | :--- | :--- | :--- |
| **Geometry** | Length | $L_{car}$ | 4.6 | m |
| | Width | $W_{car}$ | 1.6 | m |
| | Wheelbase | $L$ | 3.0 | m |
| **Kinematics** | Max Speed | $v_{max}$ | 8.0 | m/s |
| (Ackermann) | Max Accel | $a_{max}$ | 8.0 | m/s² |
| | Max Steering | $\delta_{max}$ | 1.0 (57.3°) | rad |
| | Max Steering Rate | $\omega_{max}$ | 2.0 | rad/s |
| **MPC Config** | Time Step | $\Delta t$ | 0.1 | s |
| | Horizon | $H$ | 10 | steps |
| | Ref Speed | $v_{ref}$ | 4.0 | m/s |
| **Sensing** | Range | $R_{max}$ | 20.0 | m |
| | FOV | $\alpha_{fov}$ | 180 | degree |
| | Rays | $N_{rays}$ | 100-200 | - |
