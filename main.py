import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import time
style.use('tableau-colorblind10')
import matplotlib.colors as mcolors
import cv2
import os
import shutil

##############################################################################################################################################

root_dir= os.getcwd()
image_folder = root_dir+"/image_folder"
GDP_data_loc = root_dir+"/GDP_data.xlsx"

# per iteration Frame's you wnat to keep in your video
interpolation_limit = 5

# How many top numbers you want to show
top_show_numbers = 6

### converting to per 10 million range 
divider= 10000000

###########################################################################################################################################################



def plot_gen_script(plot_dataframe,scores,cohort_size,colors,plot_names,dir_loc):
        fig, ax1 = plt.subplots(figsize=(16, 9))
        fig.text(.80, 0.25, 'Year {}'.format(plot_dataframe.year),
             fontsize=65, color='gray',
             ha='right', va='bottom', alpha=0.5)
        pos = np.arange(len(testNames))
        rects = ax1.barh(pos, [scores[k].rank for k in testNames],
                         align='center',
                         height=0.5,
                         tick_label=testNames, color=colors)

        ax1.set_title("{} VS {} VS {} GDP GROWTH - YEAR 1960 to 2019 \n in per 100 million".format(plot_names[-1],plot_names[-2],plot_names[-3])
                     ,fontsize=22,pad=30,color= "black", weight="bold")
                     
                     
        ax1.set_xlim([0, cohort_size*1.2])
        ax1.xaxis.set_major_locator(MaxNLocator(11))
        for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(17) 
        for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(17) 
        rect_labels = []
        for rect in rects:
            width = int(rect.get_width())
            rankStr = width
            if width < cohort_size:
                xloc = 5
                clr = 'black'
                align = 'left'
            else:
                xloc = 5
                clr = 'black'
                align = 'left'

            yloc = rect.get_y() + rect.get_height() / 2
            label = ax1.annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
                                textcoords="offset points",
                                ha=align, va='center',
                                color=clr, clip_on=True)
            rect_labels.append(label)
        plt.savefig(dir_loc+"/"+str(i)+".png")
        plt.close('all')
################################################################################################################################################################

base_data=pd.read_excel(GDP_data_loc,index_col=[0])
for col in base_data.columns:
    base_data[col] = pd.to_numeric(base_data[col], errors='coerce')

# GLOBAL CONSTANTS
data_vars= namedtuple('GDP', ['year'])
Score = namedtuple('Score', ['rank'])
base_data=base_data.T.bfill().ffill().T


for i in base_data.index:
    last_year_gdp = base_data[base_data.columns[-1]][i]
    filtered_base_data = base_data[base_data[base_data.columns[-1]] < last_year_gdp]
    my_base_file = filtered_base_data.dropna().iloc[:top_show_numbers,:]
    
    # Plot if there are given no of countries below the gdp of given country
    if my_base_file.shape[0] >= top_show_numbers:
        df1 = my_base_file
        ########################################################################################################################
        
        ##### Generating color list for ploting ###
        colour_list = list(mcolors.CSS4_COLORS.keys())
        colour_list =[i for i in colour_list if 'white' not in i ]
        colours=['orange','Green','Red','Yellow'] + colour_list
        ### converting to per 10 million range 
        df1=df1 / divider
        ##########################################################################################################################
        #### generating interpolated datasets for more ganurlarity
        md=pd.DataFrame(columns=['Year']+list(df1.T.columns))

        for i in range(df1.T.shape[0]-1):
            for x in range(interpolation_limit):
                if x==0:
                    append_data = dict(zip(['Year']+list(df1.T.columns),np.append(df1.T.index[i],df1.T.iloc[i,:].values)))

                else:
                    append_data = dict(zip(['Year']+list(df1.T.columns),np.append(df1.T.index[i],np.full(md.shape[1]-1, np.nan))))
                md=md.append(append_data,ignore_index=True) 

        ### converting to numerical columns as the data still will be in object form 
        for col in md.columns:
            if col != "Year":
                md[col] = pd.to_numeric(md[col], errors='coerce')

        # Linear interpolatio for generating data on more gnallure label
        md=md.interpolate(method='linear')
        md.index=md.Year
        md=md.drop(['Year'],1)
        df1=md.T
        ########################################################################################################################
        plot_names = list(md.columns)
        print(plot_names)
        colour_dict=dict(zip(df1.index,colours[:len(df1.index)]))

        for i in range(df1.shape[1]):
            print("Genearting plot >>>>>>>>>>  {}".format(i))
            testNames = list((df1.iloc[:,i].sort_values()).index)
            selected_data_vars= data_vars(df1.columns[i])
            
            ## giving the index values here
            scores = dict(zip(testNames,
                              (Score(p) for p in df1.iloc[:,i].sort_values().values)))

            ## getting the respective colours of the index here
            colours =[colour_dict[i] for i in list((df1.iloc[:,i].sort_values()).index)]

            ## giving the max value of index for that time here
            cohort_size =  max(df1.iloc[:,i].values)

            plot_gen_script(selected_data_vars,scores,cohort_size,colours,plot_names,image_folder)

        print("image generation completed")
        ########################################################################################################################
        print("generating video out of images")
        testNames = list((df1.iloc[:,1].sort_values()).index)
        video_name = '{} VS {} VS {} GDP GROWTH - YEAR 1960 to 2019.avi'.format(testNames[-1],testNames[-2],testNames[-3])


        image = [img for img in os.listdir(image_folder) if img.endswith(".png")]

        images=[str(i)+".png" for i in range(len(image))]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 10, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()