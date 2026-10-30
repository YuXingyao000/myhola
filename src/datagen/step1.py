"""Step 1：制作独立场景。保留原模块位置，兼容已有运行记录与续跑校验。"""

if __package__:
    from .room_workflow import main
else:
    from room_workflow import main


if __name__ == "__main__":
    main()
