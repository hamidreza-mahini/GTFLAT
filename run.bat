@echo on 
setlocal enabledelayedexpansion
call C:\Users\user\anaconda3\Scripts\activate.bat 

set goal[0]=Mnist-alpha0.05-ratio1
set goal[1]=FashionMnist-alpha0.05-ratio1
set goal[2]=EMnist-alpha0.05-ratio1
set goal[3]=Cifar10-alpha0.05-ratio1

set goal[4]=Mnist-alpha1.0-ratio1
set goal[5]=FashionMnist-alpha1.0-ratio1
set goal[6]=EMnist-alpha1.0-ratio1
set goal[7]=Cifar10-alpha1.0-ratio1



set g[0]=201
set g[1]=201
set g[2]=201
set g[3]=201
set g[4]=201
set g[5]=201
set g[6]=201
set g[7]=201

set l[0]=20
set l[1]=20
set l[2]=20
set l[3]=20
set l[4]=20
set l[5]=20
set l[6]=20
set l[7]=20


set algs[0]=FedAvg
set algs[1]=FedProx
set algs[2]=FedGen
set algs[3]=FedEnsemble

set t=5
set t_check=4

for /L %%j in (0, 1, 3) do (
	for /L %%i in (0, 1, 7) do (
		echo "E:\myCode\GTFLAT\!goal[%%i]!\!goal[%%i]!_!algs[%%j]!_!t_check!.h5" 
		if not exist "E:\myCode\GTFLAT\!goal[%%i]!\!goal[%%i]!_!algs[%%j]!_!t_check!.h5" (
			%CONDA_PYTHON_EXE%  E:\myCode\GTFLAT\main.py --dataset !goal[%%i]! --algorithm !algs[%%j]! --gtflat 0 --num_glob_iters !g[%%i]! --local_epochs !l[%%i]! --result_path !goal[%%i]! --times !t! 
		)
		%CONDA_PYTHON_EXE%  E:\myCode\GTFLAT\main_plot.py --dataset !goal[%%i]! --algorithms !algs[%%j]! --num_glob_iters !g[%%i]! --local_epochs !l[%%i]! --result_path !goal[%%i]! --times !t! 	
		echo "E:\myCode\GTFLAT\!goal[%%i]!\!goal[%%i]!_GT!algs[%%j]!_!t_check!.h5" 
		if not exist "E:\myCode\GTFLAT\!goal[%%i]!\!goal[%%i]!_GT!algs[%%j]!_!t_check!.h5" (
			%CONDA_PYTHON_EXE%  E:\myCode\GTFLAT\main.py --dataset !goal[%%i]! --algorithm GT!algs[%%j]! --gtflat 1 --num_glob_iters !g[%%i]! --local_epochs !l[%%i]! --result_path !goal[%%i]! --times !t! 
		)
		%CONDA_PYTHON_EXE%  E:\myCode\GTFLAT\main_plot.py --dataset !goal[%%i]! --algorithms GT!algs[%%j]! --num_glob_iters !g[%%i]! --local_epochs !l[%%i]! --result_path !goal[%%i]! --times !t! 	
	)
)

pause