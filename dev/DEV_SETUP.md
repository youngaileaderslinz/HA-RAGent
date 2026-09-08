# Setup development environment
### 1. Setup:
1. Open the terminal
2. Go to the repository folder "dev"
3. Execute ```docker compose up -d```
4. This should create the following containers:
    - home assistant
    - ollama
    - mongodb
5. Go to "http://localhost:8123" and setup your user

### 2. Install HACS:
1. Open the shell of the home assitant container
2. Execute ```wget -O - https://get.hacs.xyz | bash -``` 
3. Restart the home assistant container
4. In the home assistant web go to the "Settings/Devices & services" and click on "Add integration"
5. Select "HACS"

### 3. Create dummy entities:
1. In the home assistant web ui go to "HACS" in the sidebar
2. Search for "Virtual Components" and install the integration
3. Restart the home assistant container
4. Copy the file "dev/devices/virtual.yaml" into "dev/containers/config/"
5. Go to the "Settings/Devices & services" and click on "Add integration"
6. Select "Virtual Components" and click "Submit" and then "Skip and finish"
7. Go to the "dev/devices" folder and execute ```pip3 install -r requirements.txt``` and ```python3 setup_devices.py``` ([creating long lived access tokens](https://community.home-assistant.io/t/how-to-get-long-lived-access-token/162159/4)). The script creates the development areas and floors, assigns devices based on their names, adds searchable aliases to every entity and exposes every entity to Assist.

### 4. Download ollama models
1. In order to allow for proper test execution download the following models:
```sh
ollama pull all-minilm:33m
ollama pull qwen3:1.7b
```
