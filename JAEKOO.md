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
    python parlai/chat_service/services/browser_chat/client.py --port 10001 --serving_port 43301 --host 0.0.0.0
    # Connect to the port locally
    ssh -p 43301:localhost:43301 ...
    # Connect via i-scream
    http://i-scream.keyble.co.kr/43301
    # Launch the browser!
    # localhost:10001
    ```
    
- blenderbot2 3B 
    ```bash
    # Activate search server
    #python -u /home/jkang/project/ParlAI_SearchEngine/search_server.py serve --host 0.0.0.0:10002
    nohup sh search_server.sh > search_server.log &
    # Activate chat server
    python parlai/chat_service/services/browser_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config_blenderbot2.yml --port 10003
    # Activate client server
    python parlai/chat_service/services/browser_chat/client.py --port 10003 --serving_port 6024 --host 0.0.0.0
    # Connect to the port locally
    # Launch the browser!
    ssh ...
    # Or run ngrok
    nohup ngrok http 8080 --log=stdout > ngrok.log &
    ```

- blenderbot2 400M
    ```bash
    # Activate search server
    #python -u /home/jkang/project/ParlAI_SearchEngine/search_server.py serve --host 0.0.0.0:10002
    nohup sh search_server.sh > search_server.log &
    # Activate chat server
    python parlai/chat_service/services/browser_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config_blenderbot2_400M.yml --port 10003
    # Activate client server (with search server)
    python parlai/chat_service/services/browser_chat/client.py --port 10003 --serving_port 6024 --host 0.0.0.0
    # Connect to the port locally
    # Launch the browser!
    ssh ...
    # Or run ngrok
    nohup ngrok http 8080 --log=stdout > ngrok.log &
    ```

- blenderbot2 400M (without web search)
    ```bash
    # Activate chat server
    python parlai/chat_service/services/browser_chat/run.py --config-path parlai/chat_service/tasks/chatbot/config_blenderbot2_400M_noweb.yml --port 10003
    # Activate client server (with search server)
    python parlai/chat_service/services/browser_chat/client.py --port 10003 --serving_port 42302 --host 0.0.0.0
    # Connect to the port locally
    # Launch the browser!
    ssh ...
    # Or run ngrok
    nohup ngrok http 8080 --log=stdout > ngrok.log &
    ```


## HTML file
- /home/jkang/project/ParlAI/parlai/scripts/interactive_web.py: "WEB_HTML"

## See:
- https://parl.ai/docs/tutorial_chat_service.html#browser

---
- 2021-08-10
    - tested with blenderbot1 3B with persona manipulation (hosted)
- 2021-08-11
    - tested with blenderbot2 with internet search (hosted)