timeout 5
set root=C:\Users\SDT-SHPNP\miniconda3
call %root%\Scripts\activate.bat %root%

call conda activate base
call cd C:\Users\SDT-SHPNP\Workspace\ad_inference
call python serial_read.py

pause