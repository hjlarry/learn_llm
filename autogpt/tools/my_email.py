def send_email(
        to: str,
        subject: str,
        body: str,
        cc: str = None,
        bcc: str = None,
) -> str:
    """给指定的邮箱发送邮件"""
    print("模拟发邮件操作")
    print(f"发送内容：{body}")
    print(f"发给：{to}")
    return f"状态: 成功\n备注: 已发送邮件给 {to}, 标题: {subject}"
