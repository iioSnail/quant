"""
发送邮件
"""

from smtplib import SMTP_SSL
from email.mime.text import MIMEText


def send_mail(title, content, to_addr='iioSnail@163.com'):
    # 填写真实的发邮件服务器用户名、密码
    sender_show = "Quant系统"  # 发件人
    recipient_show = "iioSnail"  # 收件人

    user = ''  # FIXME
    password = ''  # FIXME
    # 邮件内容
    msg = MIMEText(content, 'plain', _charset="utf-8")
    # 邮件主题描述
    msg["Subject"] = title
    # 发件人显示，不起实际作用
    msg["from"] = sender_show
    # 收件人显示，不起实际作用
    msg["to"] = recipient_show
    with SMTP_SSL(host="smtp.163.com", port=465) as smtp:
        # 登录发邮件服务器
        smtp.login(user=user, password=password)
        # 实际发送、接收邮件配置
        resp = smtp.sendmail(from_addr=user, to_addrs=to_addr.split(','), msg=msg.as_string())

        print(resp)

if __name__ == '__main__':
    message = 'Python 测试邮件...'
    Subject = '主题测试'
    # 实际发给的收件人
    send_mail(message, Subject)
