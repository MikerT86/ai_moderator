import os
import argparse
import tqdm
import pandas as pd
from irc_twitch import Collector


#[TODO]: Use argument - LOG_DIR instead of current string
# logs_dir = '../.chatty/logs/'
# channels = ['pgl']


class DataLoader:
    def __init__(self, log_dir: str) -> None:
        if os.path.exists(log_dir):
            self.log_dir = log_dir 
        else:
            raise FileExistsError(f"{log_dir}: not exists! Initiate another log directory!")

        self.log_files = self.get_log_files()

    def __info__(self):
        print('\n----- Data Loader Settings -----')
        print(f"Logs Directory: {self.log_dir}")
        print(f"Log Files: {self.log_files}")

    def read_log(self, channel):
        filename = self.log_dir + channel + '.log'
        try:
            with open(filename, 'r') as log_file:
                data = log_file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f'There is no log from channel: {channel}')

        return data

    def get_log_files(self):

        return [files[0] for _, _, files in os.walk(self.log_dir)]


def mark_banned_message(ban_chunks, df):

    for chunk in ban_chunks:
        if chunk.startswith("("):
            continue
        user_messages = df.loc[df.user == chunk]
        if user_messages.shape[0]:
            message_index = user_messages.tail(1).index[0]
            df.at[message_index, 'banned'] = 1


def process_log_file_data(data):

    user_marker_start = '<'
    user_marker_end = '> '

    timestamp_marker_start = '['
    timestamp_marker_end = '] '

    df_full_data = pd.DataFrame(columns=['timestamp', 'user', 'message', 'banned'])
    for idx_line in tqdm.trange(len(data)):
        line = data[idx_line]
        if 'BAN: ' in line:
            ban_chunks = line.split()[3:]
            mark_banned_message(ban_chunks, df_full_data)
        if '<' in line:
            user = line[line.find(user_marker_start) + 1:line.find(user_marker_end)]
            timestamp = line[line.find(timestamp_marker_start) + 1:line.find(timestamp_marker_end)]
            message = line[line.find(user_marker_end) + 2:-1]

            df_full_data = df_full_data.append({'timestamp': timestamp,
                                                'user': user.replace('+', ''),
                                                'message': message,
                                                'banned': 0}, ignore_index=True)

    return df_full_data


def main(params):

    dl = DataLoader(params.log_dir)
    dl.__info__()

    if params.regime == 'offline':
        log_data = dl.read_log(f'#{params.channel}')
        df_data = process_log_file_data(log_data)
    else:
        collector = Collector(params.token)
        collector.connect(params.channel)
        df_data = collector.collect_messages()

    print("\n----Data Stats------")
    print("Shape: ", df_data.shape)
    print("Banned messages: ", df_data.banned.sum())
    df_data.to_csv('pgl_chat_data.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", default="./chatty_logs/",
                        help="Initiate logs directory", type=str)
    parser.add_argument("-c", "--channel", default="pgl",
                        help="Initiate channels", type=str)
    parser.add_argument("-r", "--regime", default="offline",
                        help="Choose regime of data load (offline\online)", type=str)
    parser.add_argument("-u", "--username", default="mikert86",
                        help="Username", type=str)
    parser.add_argument("-t", "--token", default="oauth:apqd0s2x8wz81mqs3ju3jgh51so09c",
                        help="Insert you personal twitch account OAuth token", type=str)
    
    args = parser.parse_args()
    main(args)

