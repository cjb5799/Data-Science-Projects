
#-
#author: "Carla Bradley"
#date: "6/23/2021"

library(mlbench)
library(caret)
library(car)  
library(lattice)
library(dplyr)
library(ggplot2)
library(tidyverse)
#install.packages("d3heatmap")


setwd("/Users/carla/Desktop/Summer 2021/DSC630 Predictive/Week 3")
data <- read.csv("dodgers.csv")


# Organize day-of-week variable
data$ordered_day_of_week <- with(data=data,
  ifelse ((day_of_week == "Monday"),1,
  ifelse ((day_of_week == "Tuesday"),2,
  ifelse ((day_of_week == "Wednesday"),3, 
  ifelse ((day_of_week == "Thursday"),4,
  ifelse ((day_of_week == "Friday"),5,
  ifelse ((day_of_week == "Saturday"),6,7)))))))
data$ordered_day_of_week <- factor(data$ordered_day_of_week, levels=1:7,
labels=c("Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"))



# Order month variable for plots and data summaries
# 
data$ordered_month <- with(data=data,
  ifelse ((month == "APR"),4,
  ifelse ((month == "MAY"),5,
  ifelse ((month == "JUN"),6,
  ifelse ((month == "JUL"),7,
  ifelse ((month == "AUG"),8,
  ifelse ((month == "SEP"),9,10)))))))
data$ordered_month <- factor(data$ordered_month, levels=4:10,
labels = c("April", "May", "June", "July", "Aug", "Sept", "Oct"))

                              
                            
# Exploratory data analysis


# Graph display attendance by day of week

with(data=data,plot(ordered_day_of_week, attend/1000, 
xlab = "Day of Week", ylab = "Attendance (thousands)", 
col = "pink", las = 3))

with(data, table(bobblehead,ordered_day_of_week)) 
# Bobbleheads are given mostly on Tuesdays
# 

# Attendance vs conditioning on day/night
# skies and whether  fireworks are displayed
#  graphical summary 
group.labels <- c("No Fireworks","Fireworks")
group.symbols <- c(21,24)
group.colors <- c("blue","blue") 
group.fill <- c("blue","red")

xyplot(attend/1000 ~ temp | skies + day_night, 
    data = data, groups = fireworks, pch = group.symbols, 
    aspect = 1, cex = 1.5, col = group.colors, fill = group.fill,
    layout = c(2, 2), type = c("p","g"),
    strip=strip.custom(strip.levels=TRUE,strip.names=FALSE, style=1),
    xlab = "Temperature", 
    ylab = "Attendance by thousands",
    key = list(space = "top", 
        text = list(rev(group.labels),col = rev(group.colors)),
        points = list(pch = rev(group.symbols), col = rev(group.colors),
        fill = rev(group.fill))))          

#The graph  shows that  the  attendance also increases when there are fireworks  which is a great day  for a promotional announcement.

        
 
# Evaluating multiple  model to identify the appropriate.
# 
# 
control <- trainControl(method="repeatedcv", number=10, repeats=3)

#Linear regression
set.seed(7)
fit.lm <- train(attend~., data=data, method="lm", trControl=control)

# CART
set.seed(7)
fit.cart <- train(attend~., data=data, method="rpart", trControl=control)
# LDA
set.seed(7)
fit.lda <- train(attend~., data=data, method="lda", trControl=control)
# SVM
set.seed(7)
fit.svm <- train(attend~., data=data, method="svmRadial", trControl=control)
# kNN
set.seed(7)
fit.knn <- train(attend~., data=data, method="knn", trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(attend~., data=data, method="rf", trControl=control)

# collect re-samples KNN - RF -LM Modeles only
results <- resamples(list(KNN=fit.knn, RF=fit.rf, LM=fit.lm))
summary(results)


# box and whisker plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)


# density plots of accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
densityplot(results, scales=scales, pch = "|")
#shows the distribution of model accuracy as density plots.

# dot plots of accuracy
scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(results, scales=scales)

#pair-wise scatter plots of predictions to compare models
splom(results)
#the graphs shows a strong correlation between LM  and KNN compared to other models


# xyplot plots to compare models
xyplot(results, models=c("KNN", "RF","lm"))



#  Linear Training-and-test  model has been selected for accuracy 
set.seed(1234)

training_test <- c(rep(1,length=trunc((2/3)*nrow(data))),
rep(2,length=(nrow(data) - trunc((2/3)*nrow(data)))))


data$training_test <- sample(training_test) # random permutation 
data$training_test <- factor(data$training_test, 
  levels=c(1,2), labels=c("TRAIN","TEST"))
data.train <- subset(data, training_test == "TRAIN")

# check training data frame
print(str(data.train))
data.test <- subset(data, training_test == "TEST")

# check test data frame
print(str(data.test)) 



# specify a simple model with bobblehead 
my.model <- {attend ~ ordered_month + ordered_day_of_week + bobblehead}


# fit the training model 
train.model.fit <- lm(my.model, data = data.train)


# summary of model fit to the training set
print(summary(train.model.fit))

# training set predictions from the model fit to the training set
data.train$predict_attend <- predict(train.model.fit) 


# test set predictions from the model fit to the training set
data.test$predict_attend <- predict(train.model.fit, 
  newdata = data.test)

# compute the proportion of response variance
# accounted for when predicting out-of-sample
cat("\n","Proportion of Test Set Variance  : ",
round((with(data.test,cor(attend,predict_attend)^2)),
  digits=3),"\n",sep="")

# merge the training and test sets for plotting
data.plotting.frame <- rbind(data.train,data.test)

# generate predictive modeling visual 
group.labels <- c("No Bobbleheads","Bobbleheads")
group.symbols <- c(21,24)
group.colors <- c("blue","red") 
group.fill <- c("blue","red")  
xyplot(predict_attend/1000 ~ attend/1000 | training_test, 
       data = data.plotting.frame, groups = bobblehead, cex = 2,
       pch = group.symbols, col = group.colors, fill = group.fill, 
       layout = c(2, 1), xlim = c(20,65), ylim = c(20,65), 
       aspect=1, type = c("p","g"),
       panel=function(x,y, ...)
            {panel.xyplot(x,y,...)
             panel.segments(25,25,60,60,col="black",cex=2)
            },
       strip=function(...) strip.default(..., style=1),
       xlab = "Actual Attendance by thousands)", 
       ylab = "Predicted Attendance by thousands)",
       key = list(space = "top", 
              text = list(rev(group.labels),col = rev(group.colors)),
              points = list(pch = rev(group.symbols), 
              col = rev(group.colors),
              fill = rev(group.fill))))     


# use data set to obtain an estimate of the increase in
# attendance due to bobbleheads, controlling for other factors 
my.model.fit <- lm(my.model, data = data)  # use all available data
print(summary(my.model.fit))


# tests statistical significance of the bobblehead promotion
# 
print(anova(my.model.fit))  
cat("\n","Estimated Effect of Bobblehead Promotion on Attendance: ",
round(my.model.fit$coefficients[length(my.model.fit$coefficients)],
digits = 0),"\n",sep="")

# Bobblehead Promotion days are the best day to  release new promotional offer due to the high attendance on the game.












