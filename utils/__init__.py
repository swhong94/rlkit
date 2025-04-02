
from .logger import Logger 
from .replay_buffer import ReplayBuffer 


__all__ = ["Logger", "ReplayBuffer"] 



# import os 
# import logging 
# import csv 
# from datetime import datetime  

# import numpy as np 
# import matplotlib.pyplot as plt 

# def rolling_average(data, window=100): 
#     """
#     Compute rolling average of the data (add dummies before real data)
#     """
#     data = np.pad(data, (window-1, 0), mode='constant', constant_values=data[0])
#     return np.convolve(data, np.ones(window)/window, mode='valid') 

# class Logger: 
#     """
#     Logger class for logging training information
#     """
#     def __init__(self, 
#                  log_dir, 
#                  log_name_prefix="training",
#                  plot_window=None):
#         os.makedirs(log_dir, exist_ok=True) 
#         self.timestamp = timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         self.log_name_prefix = log_name_prefix 
#         log_file = os.path.join(log_dir, f"{log_name_prefix}_{timestamp}.log")

#         self.logger = logging.getLogger(log_name_prefix) 
#         self.logger.setLevel(logging.INFO) 

#         fh = logging.FileHandler(log_file) 
#         fh.setLevel(logging.INFO) 

#         ch = logging.StreamHandler() 
#         ch.setLevel(logging.INFO) 

#         formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")      # Define the format of the log message 
#         fh.setFormatter(formatter) 
#         ch.setFormatter(formatter) 

#         self.logger.addHandler(fh) 
#         self.logger.addHandler(ch) 

#         # Set CSV setup 
#         self.csv_file = os.path.join(log_dir, f"{log_name_prefix}_{timestamp}.csv")
#         self.csv_writer = None 
#         self.csv_header_written = False 

#         # Plotting setup 
#         self.plot_window = plot_window 
#         self.episodes = [] 
#         self.metrics_history = {} 
#         self.fig = None 
#         self.axes = {} 
#         self.lines = {} 
#         self.smoothed_lines = {} 


#     def info(self, message): 
#         self.logger.info(message) 

#     def log_metrics(self, episode, metrics): 
#         """Log structured metrics to CSV file, appending each time"""
#         with open(self.csv_file, "a", newline="") as f: 
#             csv_writer = csv.writer(f) 
#             if not self.csv_header_written: 
#                 csv_writer.writerow(['episode'] + list(metrics.keys()))
#                 self.csv_header_written = True 
#             row = [episode] + list(metrics.values()) 
#             csv_writer.writerow(row) 

#         # Update metrics history 
#         self.episodes.append(episode) 
#         for key, value in metrics.items(): 
#             if key not in self.metrics_history: 
#                 self.metrics_history[key] = [] 
#             self.metrics_history[key].append(value) 

#         # Initialize plotting on first call when metrics are known 
#         if self.fig is None: 
#             plt.ion() 
#             num_metrics = len(metrics) 
#             self.fig, axes = plt.subplots(num_metrics, 1, figsize=(10, num_metrics*3)) 
#             if num_metrics == 1: 
#                 axes = [axes] 
#             plt.xlabel("Episode") 
#             self.fig.suptitle(f"{self.logger.name.capitalize()} Training Progress")
#             for i, key in enumerate(metrics.keys()): 
#                 self.axes[key] = axes[i]
#                 self.axes[key].set_ylabel(key.replace("_", " ").title())


#         # Compute rolling average 
#         if self.plot_window is not None: 
#             # Adjust transparency of original lines 
#             smoothed_metrics = {
#                 key: rolling_average(np.array(self.metrics_history[key]), window=self.plot_window)
#                 for key in self.metrics_history.keys() 
#             }
#         alpha = 1.0 if self.plot_window is None else 0.3 

        
#         for key in metrics.keys():
#             if key not in self.lines: 
#                 self.lines[key], = self.axes[key].plot(self.episodes, self.metrics_history[key], 
#                                                        label=f"{key.replace('_', ' ').title()}", alpha=alpha)
#                 if self.plot_window is not None: 
#                     self.smoothed_lines[key], = self.axes[key].plot(self.episodes, smoothed_metrics[key], 
#                                                                    label=f"{key.replace('_', ' ').title()} (MA={self.plot_window})")
#             else: 
#                 self.lines[key].set_data(self.episodes, self.metrics_history[key])
#                 if self.plot_window is not None: 
#                     self.smoothed_lines[key].set_data(self.episodes, smoothed_metrics[key])
            

#             self.axes[key].relim()
#             self.axes[key].autoscale_view() 
#             self.axes[key].legend() 

#         plt.tight_layout() 
#         plt.draw() 
#         plt.pause(0.01) 

#     def close(self): 
#         for handler in self.logger.handlers[:]: 
#             handler.close() 
#             self.logger.removeHandler(handler) 
        
#         plt.ioff() 
#         # Save plot to results directory
#         if not os.path.exists("results"):
#             os.makedirs("results")
#         plt.savefig(os.path.join("results", f"{self.log_name_prefix}_{self.timestamp}.png"))
#         plt.close(self.fig) 
    