# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:00:40 2020

@author: kxj17699
"""

import math
import pandas as pd
import numpy as np

from scipy import stats
import matplotlib.pyplot as plt



class univariate_distribution:
    '''
    Class to create empirical univariate distributions
    Input = numpy vector/Pandas column
    Outputs = PDF, CDF & combination function and histogram plot
    Functions = evaluate the PDF for a given X value, integrate the CDF for a given X value (output probability)
    
    obtain pdf value for a given x-value
    def evaluate_pdf(self, X_val):
    
    obtain probability for a given x-value based upon the caslculated cdf
    integrate_cdf(self, X_val):
    
    obtain x value for a given percentile
    def kde_percentile(self, percentile_val):
    
    creates output plot
    def kdeplot(self,label='')
    
    K JAGGS DEC 2020
    '''
    
    
    def __init__(self):
        #instantiate class
        print("\nUnivariate_distribution class initialised")

    def univariate_distribution_calculation(self, X_in,kde_bandwidth =0):
        
        #input data vector
        self.X_in = X_in
        
        #defaul kde calculation or user specified
        self.kde_bandwidth = kde_bandwidth
        #dataset size prior to removing nan & inf values
        print("Input dataset size =", self.X_in.shape)        
        
        #remove any NaN values from numpy array - these values will trigger errors later
        self.vector_remove_nan()
        #print updated dataset size after removal of nan or inf values
        print("Conditioned dataset size =",self.X_in.shape)
        
        #create the distribution
        self.univariate_kde()
 
    def univariate_kde(self):
        '''
        (int)->
        Method to implement Kernel Density Estimation to estimate a probability density function
        Uses scipy stats gaussian_kde
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
        Option to specify a bandwidth before declaration using kde_bandwidth. Default is 0 and uses the default Kernel bandwidth settings.
        kJAGGS MARCH 2022
        '''
        
        #create the pdf function - currently uses scott's method for calculating bandwidth
        if self.kde_bandwidth == 0:
            self.kde = stats.gaussian_kde(self.X_in)
        else:
            self.kde = stats.gaussian_kde(self.X_in, bw_method= self.kde_bandwidth)

        #create numpy array of x values based upon input min and input max divided into 1000 increments
        self.pdf_grid()

        #create cdf of output pdf plot from gaussian_kde
        self.cdf = [self.kde.integrate_box_1d(-math.inf,x) for x in self.univariate_x]

    def vector_remove_nan(self):
        '''
        function to remove nan & inf values from an array before calculating univariate distribution
        Some processes will return error if NaN is present in vector series
        PRECONDTION: single vector array, spurious values are set to np.NaN
        (Pandas Dataframe)(column name as string)->(numpy vector array)
        Example input X = [1,2,3,4,NaN,6,7,8,9]
        output X = [1,2,3,4,6,7,8,9]
        '''    
     
        self.X_in = self.X_in[(np.isnan(self.X_in) == False) & (np.isinf(self.X_in) == False)]


    def pdf_grid(self):
        '''
        Function to create a numpy array for fitting the x-axis of a pdf function
        (numpy vector array)=>(numpy vector array)
        PRECONDITION data has been filtered/qcd/no nans
        Example X Array min = 201, X Array max = 300 => [201,202,203...300] 
        K JAGGS Sep 2018
        '''

        self.univariate_x =  np.linspace(np.min(self.X_in), np.max(self.X_in), 1000)
        

    def evaluate_pdf(self, X_val):
        '''
        (float)->(float)
        Evaluate the pdf for a given X-value
        '''

        return self.kde.evaluate(X_val)[0]
        
    def integrate_cdf(self, X_val):
        '''
        (float)->(float)
        Return the probability of the CDF for a given X value
        '''


        return self.kde.integrate_box_1d(-math.inf,X_val)

    def kde_descriptive_stats(self):
        '''
        Descriptive statistics of the input dataset and distributions

        '''
        print("\nNumber of data points =", self.X_in.shape[0] )
        print("\nMean =", np.mean(self.X_in) )
        print("Mode =", float(stats.mode(self.X_in)[0]) )
        print("Variance =", np.var(self.X_in) )
        print("Std Dev =", np.std(self.X_in) )
        
        print("\nPmin =", np.min(self.X_in) )
        print("P10 =", self.kde_percentile(10))
        print("P50 =", self.kde_percentile(50))
        print("P90 =", self.kde_percentile(90))
        print("Pmax =", np.max(self.X_in))

    def kde_percentile(self, percentile_val):
        '''
        (int)->(float)
        Calculate the selected percentile of the input dataset 
        '''
        return np.percentile(self.X_in, percentile_val)

    def kdeplot(self,label=''):
        '''
        Final plot of data
        Combination of histogram and pdf + cdf (different axes)
        User option to add label
        KJAGGS DEC 2020
        '''
       
        self.fig_hist, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.univariate_x, self.kde.evaluate(self.univariate_x), linewidth=3, alpha=0.5, c="red", label = "PDF")
        #, label='bw=%.2f' % self.kde.bandwidth
        
        ax2.plot(self.univariate_x, self.cdf, linewidth=3, alpha=0.5, c="blue", label='CDF')
        ax1.hist(self.X_in, 20, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
        ax1.set_title("PDF & CDF of input data :" + label)
        ax1.set_ylabel("PDF")
        ax1.set_xlabel(label)
        ax2.set_ylabel("CDF")
        
        plt.axvline(x=self.kde_percentile(10),color='red')
        plt.axvline(x=self.kde_percentile(50),color='green')
        plt.axvline(x=self.kde_percentile(90),color='blue')
        plt.axvline(x=np.mean(self.X_in),label = 'mean',color='black', linestyle = '--')
        
        self.fig_hist.legend(loc='right')
        
        #label offset = 1.25% of the plotted x range.
        #label is plotted - 0.0125
        self.label_offset = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.0125
        #print(self.label_offset)
        

        #plot text boxes
        trans = ax1.get_xaxis_transform()
        plt.text(self.kde_percentile(10)-self.label_offset, 0.8,'P90',color='red',size=10, va =  'center' , ha =  'center', bbox=dict(facecolor='white', edgecolor='red',alpha=1,pad= 0.25,boxstyle="round"),transform=trans,rotation=90)
        plt.text(self.kde_percentile(50)-self.label_offset,0.8, 'P50',color='green',size=10, va =  'center' , ha =  'center', bbox=dict(facecolor='white', edgecolor='green',alpha=1,pad= 0.25,boxstyle="round"), transform=trans,rotation=90)
        plt.text(self.kde_percentile(90)-self.label_offset,0.8, 'P10',color='blue',size=10, va =  'center' , ha =  'center', bbox=dict(facecolor='white', edgecolor='blue',alpha=1,pad= 0.25,boxstyle="round"), transform=trans,rotation=90)


if __name__ == "__main__":
    
    ####################################################
    #EXAMPLE WORKFLOW
    
    #read example dataset
    inpath = "O:\\Asset_Mangmt\\ClipperSouth\\5_G&G_Petrophysics\\Geophysics\\2021\\KJ CLIPPER SE\\Clipper SE Depth Conversion Review May 2021\\Bayesian Kriging\\Review\\Bayesian Kriging Statistics FWL CSV.csv"
    df = pd.read_csv(inpath, encoding = 'utf-8',header=[0]) 
    #print(df.columns)
    
    #initialise univariate distribution class
    ud= univariate_distribution()
    
    #calculate statistics - input must be pandas column or numpy array
    ud.univariate_distribution_calculation(df['Vol > contact'])    
    #print(ud.evaluate_pdf(200))
    
    #calculate probability from cdf of a given x value
    print(ud.integrate_cdf(6670200000))
    
    #output plot for distribution - user specified title
    ud.kdeplot(label = "GRV above FWL 200ft standoff")
    
    ud.kde_descriptive_stats()
    
        
