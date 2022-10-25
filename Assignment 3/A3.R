
library(ggplot2)

#Task 4
#read data
rm(list = ls())
setwd("C:/A3/TaskB")
df <- read.csv("Donald.csv")
str(df)

#turn date from char to date
df[['date']]<-strptime(df[['date']], 
                       format = "%a %b %e %H:%M:%S %z %Y", 
                       tz = "GMT")
df[['date']]<-as.POSIXct(df[['date']])
str(df)

ggplot(df, 
       aes(x=date)) + 
  labs(
    title = "Post over the time",
    x = "Time",
    y = "Post Count")+
  geom_histogram(bins = 100,
                 fill="#69b3a2", 
                 color="#e9ecef", 
                 alpha=0.9)

#Task 5
#read data
df2 <- read.csv("user.csv")
str(df2)
p = ggplot(df2, 
       aes(x=count)) + 
  labs(
    title = "Distribution of the number of Tweets",
    x = "Post Count",
    y = "Account Count")+
  geom_histogram(bins = 100,
                 fill="#69b3a2", 
                 color="#e9ecef", 
                 alpha=0.9)
p
p + coord_cartesian( xlim = c(50,150),ylim = c(0,100))
p + coord_cartesian( xlim = c(15,50),ylim = c(100,10000))
