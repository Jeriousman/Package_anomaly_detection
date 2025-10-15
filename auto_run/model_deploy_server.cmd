set root=C:\Users\SDT-SHPNP\miniconda3
call %root%\Scripts\activate.bat %root%

call conda activate base
call cd C:\Users\SDT-SHPNP\Workspace\model_deploy
call uvicorn main:app --reload --host 0.0.0.0

pause