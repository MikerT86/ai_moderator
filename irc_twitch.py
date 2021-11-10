import socket
import datetime
from tqdm import tqdm

OAUTH_TOKEN = 'oauth:apqd0s2x8wz81mqs3ju3jgh51so09c'
USERNAME = 'mikert86'
LOGS_DIR = './collector_logs'


class Collector:

    LOGS_DIR = './collector_logs/'

    def __init__(self, username, password):
        self.irc_server = 'irc.twitch.tv'
        self.irc_port = 6667
        self.channel = ''
        self.password = password
        self.username = username
        self.irc = socket.socket()

    def send_command(self, command):
        if 'PASS' not in command:
            print(f'< {command}')
        self.irc.send((command + '\r\n').encode())

    def connect(self, channel):
        try:
            self.channel = channel
            self.irc.connect((self.irc_server, self.irc_port))
            self.send_command(f'PASS {self.password}')
            self.send_command(f'NICK {self.username}')
            self.send_command(f'JOIN #{channel}')
            print(f'Successfully connected to channel: {channel}')
        except ConnectionError:
            raise ConnectionError

        self.collect_messages()

    def __get_user_from_msg(self, message):
        return message.split('!')[0]

    def __get_message_from_msg(self, message):

        msg_parts = message.split(':')
        if len(msg_parts) == 0:
            return ''

        if len(msg_parts) > 2:
            msg = ':'.join(msg_parts[1:])
        else:
            msg = msg_parts[1]

        return msg

    def parse_message(self, message):
        # print(f'> {message}')
        msg_tag = 'PRIVMSG'
        ban_tag = 'CLEARCHAT'
        channel, user, msg, tag = self.channel, '', '', ''

        if msg_tag in message:
            msg_array = message.split(msg_tag)
            user = self.__get_user_from_msg(msg_array[0].replace(':', ''))
            msg = self.__get_message_from_msg(msg_array[1])
            tag = msg_tag

        elif ban_tag in message:
            msg_array = message.split(ban_tag)
            user = self.__get_user_from_msg(msg_array[0].replace(':', ''))
            msg = self.__get_message_from_msg(msg_array[1])
            tag = msg_tag
        elif message != '':
            print(f'> {message}')

        return [str(datetime.datetime.now()), channel, user, msg, tag]

    def collect_messages(self, MAX_PRIVMSG=10000):
        try:
            file = open(f'{self.LOGS_DIR}/{self.channel}_log.csv', 'a')
        except FileNotFoundError:
            raise FileNotFoundError

        # file.write('timestamp,channel,user,message,msg_type')

        collected_msgs = 0
        ban_msgs = 0
        print("\n-------Session Started----------")
        while True:
            received_msgs = self.irc.recv(2048).decode()
            for message in received_msgs.split('\r\n'):
                msg_parts = self.parse_message(message)
                if msg_parts[2] != '' and msg_parts[3] != '':
                    file.write(', '.join(msg_parts)+'\n')
                    collected_msgs += 1
                    if msg_parts[-1] == 'CLEARCHAT':
                        ban_msgs += 1
            if collected_msgs == MAX_PRIVMSG:
                print("Banned users: ", ban_msgs)
                print("Collected msgs: ", collected_msgs)
                break

        file.close()


def main():
    collector = Collector(USERNAME, OAUTH_TOKEN)
    collector.connect('rainbow6')


if __name__ == '__main__':
    main()
