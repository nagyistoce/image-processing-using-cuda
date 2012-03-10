@echo off
echo "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\bin\nvcc.exe"    -gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\"  --machine 32 -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin"    -Xcompiler "/EHsc /W3 /nologo /O2 /Zi   /MT  "  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\include" -maxrregcount=32  --compile -o "Debug/gpufunctions.cu.obj" gpufunctions.cu 
           "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\bin\nvcc.exe"    -gencode=arch=compute_10,code=\"sm_10,compute_10\" -gencode=arch=compute_20,code=\"sm_20,compute_20\"  --machine 32 -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin"    -Xcompiler "/EHsc /W3 /nologo /O2 /Zi   /MT  "  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v3.2\include" -maxrregcount=32  --compile -o "Debug/gpufunctions.cu.obj" "c:\Users\toshiba\Documents\Acads\ImageProcessingV3\gpufunctions.cu" 
if errorlevel 1 goto VCReportError
goto VCEnd
:VCReportError
echo Project : error PRJ0019: A tool returned an error code from "Compiling with CUDA Build Rule..."
exit 1
:VCEnd