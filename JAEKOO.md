# How to make chat services

## Run:
- blenderbot1 90M
    ```bash
    # Activate chat server
    python parlai/chat_service/services/browser_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config.yml --port 10001
    # Activate client server
    python parlai/chat_service/services/browser_chat/client.py --port 10001
    # Connect to the port locally
    ssh ... -p 1001
    # Launch the browser!
    # localhost:10001
    ```

- blenderbot1 3B 1024 (persona)
    ```bash
    # Activate chat server
    python parlai/chat_service/services/browser_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config_blenderbot3B_persona.yml --port 10001
    # Activate client server
    python parlai/chat_service/services/browser_chat/client.py --port 10001
    # Connect to the port locally
    ssh ... -p 1001
    # Launch the browser!
    # localhost:10001
    ```
- blenderbot2 3B 
    ```bash
    # Activate search server
    #python -u /home/jkang/project/ParlAI_SearchEngine/search_server.py serve --host 0.0.0.0:10002
    nohup sh search_server.sh > search_server.log &
    # Activate chat server
    python parlai/chat_service/services/browser_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config_blenderbot2.yml --port 10001
    # Activate client server
    python parlai/chat_service/services/browser_chat/client.py --port 10001
    # Connect to the port locally
    # Launch the browser!
    ssh ...
    # Or run ngrok
    nohup ngrok http 8080 --log=stdout > ngrok.log &
    ```

- Using ngrok


## See:
- https://parl.ai/docs/tutorial_chat_service.html#browser

---
- 2021-08-10
    - tested with blenderbot1 3B with persona manipulation (hosted)
- 2021-08-11
    - tested with blenderbot2 with internet search (hosted)