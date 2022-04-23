TO RUN THE PROGRAM ON THE CPU:
- Open the cmd prompt
- Move on the project directory using "cd" command
- run the command "pip install -r requirements.txt"
- pip3 install torch torchvision torchaudio
- run the command "python main.py"

TO RUN THE MODEL ON THE NVIDIA GPU:
- Open the cmd prompt
- Move on the project directory using "cd" command
- run the command "pip install -r requirements.txt"
- To avoid malfunctions, it is suggested to uninstall some packages, running the command "uninstall torch torchvision torchaudio"
- Now you must download the CUDA Toolkit for your OS and install it. Finally, run the command "pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113"
- run the command "python main.py"
