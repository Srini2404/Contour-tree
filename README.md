# Contour-tree 
### Software Needed
* **Python 3.9+**
    1) **Windows users** - Go to python website and then download the version above 3.9. Once downloaded, run the .exe file that you just downloaded.
    2) **Ubuntu users** - Type the following in the terminal.</br>
    ```sudo apt update && sudo apt upgrade```.</br>
    Then we should install the required dependencies for them. Following is the code for the same. </br>
    ```sudo apt install wget build-essential libncursesw5-dev libssl-dev \ libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev``` </br>
    Then to download python we do the follwing</br>
    ```sudo add-apt-repository ppa:deadsnakes/ppa```</br>
    Then to install python we do the following </br>
    ```sudo apt install python3.11```</br>
* **pip**
    1) **Windows** - for installing it on windows we need python installed already in the system. Now open the terminal and type this</br>
    ```curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py```</br>
    and then followed by this.</br>
    ```python get-pip.py```</br>
    2) **Ubuntu**  - for intalling it on ubuntu for the terminal and type the below mentioned commands in present.</br>
    ```sudo apt update```</br>
    followed by </br>
    ``` sudo apt install python3-pip```</br>
* **numpy module** - for installing all the subsequent packages, pip must be installed in the system.</br>
```pip install numpy```</br>
* **collection module** - ```pip install collection```</br>
* **scipy.spatial module** - ``` pip install scipy```</br>
* **random module** - built in module </br>
* **math module** - built in module, so no need of anything else.</br>
* **matplotlib module** - ``` pip install matplotlib```</br>
* **queue module** - built in module .</br>

## Instructions to Run the code - </br>
Once all the needed packages are installed, we are all set to run the code. All we need is to now just open the SourceCode file (python file - final.py) in the editor of your choice ( we would recommend **Visual Studio Code**) and press the run button. Or download the file, and on the terminal, in the folder which contains the code file (final.py), type ```python3 [filename].py``` </br>
The data needed for running the code is hard coded in the file. If we want to change the input data we just modify the values in the file and then run the file.
Once the running is complete a plot will show up which is the **contour tree** for the given set of data points.


