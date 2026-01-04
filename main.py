from ui import ui

if __name__ == "__main__":
    print("正在启动农业决策系统 AID v2...")
    print("界面将在浏览器中打开...")

    # 创建并启动界面
    demo = ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
