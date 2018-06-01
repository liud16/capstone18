import matplotlib.pyplot as plt
import pandas as pd

def visualize(peak_dict):
    for i in range(len(peak_dict)):
        df = pd.DataFrame(peak_dict['peak_%s' % i], 
        columns=['Position', 'Height', 'Width', 'Time'])
        
        plt.subplot(3, 1, 1)
        plt.plot(df['Time'], df['Height'])
        plt.title('Peak %s Dynamics' % (i+1))
        plt.ylabel('Intensity')

        plt.subplot(3, 1, 2)
        plt.plot(df['Time'], df['Position'])
        plt.ylabel('Position')

        plt.subplot(3, 1, 3)
        plt.plot(df['Time'], df['Width'])
        plt.ylabel('Width')
        plt.xlabel('Time')
        plt.show()
    return
