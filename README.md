# bot-code

The code for controlling the robots.

To use, clone the robus-core repository and place it in this repository.
It functions as the core connecting all the nodes in this repository.
Rename the folder to robus_core.

```bash
cd bot-code
git clone https://github.com/Robocup-Junior-Open-League/robus-core
mv robus-core robus_core
```

The core can be updated independently from the nodes by navigating to the robus_core directory and pulling the latest changes.

```bash
cd bot-code/robus_core
git pull
```

Before starting, you should set up a python virtual environment and install the required packages.
The Python version should be 3.13.1.

```bash
cd bot-code
python -m venv venv
venv\Scripts\activate
```

Python libraries need to be installed to run:

```bash
cd bot-code
pip install -r requirements.txt
```

To start the bot, run the following commands:

Linux:

```bash
cd bot-code
bash ./robus_core/start.sh
```

Windows:

```bash
cd bot-code
robus_core\start.bat
```

To stop all nodes, run the following command:

Linux:

```bash
cd bot-code
bash ./robus_core/stop.sh
```

Windows:

```bash
cd bot-code
robus_core\stop.bat
```
