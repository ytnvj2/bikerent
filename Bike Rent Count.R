# Lets start with loading the data
df=read.csv('./day.csv')
# Lets see the structure of the data
str(df)
# Convert the Variables that are factors but were taken as numeric by R
df$season=as.factor(df$season)
df$yr=as.factor(df$yr)
df$mnth=as.factor(df$mnth)
df$holiday=as.factor(df$holiday)
df$weekday=as.factor(df$weekday)
df$workingday=as.factor(df$workingday)
df$weathersit=as.factor(df$weathersit)
df$dteday=as.character.Date(df$dteday)
# Giving the columns meaningful Names 
colnames(df)=c('Instant','Date','Season','Year','Month','Holiday','WeekDay','WorkingDay','WeatherSituation','Temp','ActualTemp','Humidity','WindSpeed','Casual','Registered','Count')
# Lets view the structure again
str(df)
# Lets get a quick summary of the variables in the DataFrame
summary(df)
# remove columns that are of no use
df$Instant=NULL
df$Date=NULL
cat_cols=NULL
num_cols=NULL
for(i in 1:ncol(df)){
  if(is.factor(df[,i])){
    cat_cols=c(cat_cols,i)
  }
  else{
    num_cols=c(num_cols,i)
  }
}
#df$Casual=NULL
#df$Registered=NULL
# Lets step into visualization
#  For Numerical Variables, we will be using the multi.hist fucntion to visualize
#  All variables in one go. The plots will contain each variable's histogram,
#  KDE plot, and a line representing Normal Distribution for comparison.
#   Import package from library
library(psych)
library(ggplot2)
#   Plot the multi.hist for all numeric variables in DF
multi.hist(df[,8:ncol(df)],dcol =c('blue','red'), dlty = c('solid','solid'),main = 'Variable Analysis' )
#   Now for the factors, lets plot bar graph to see the count of each class 
init=ggplot(data=df)
for(i in cat_cols){
  plot=init+geom_bar(aes(x=df[,colnames(df[,i,drop=F])]),fill='blue',colour='blue')+xlab(colnames(df[,i,drop=F]))+
    ylab('Frequency')
  print(plot)
}
#   Now lets only see the distributions of the skewed variables with their mean and median.
lh <- 10:12
bw <- c(0.05, 0.02, 200)
for (i in 1:3) {
  plot <- init +
    geom_histogram(aes(x = df[,lh[i]]), binwidth = bw[i])+
    geom_vline(aes(xintercept = mean(df[,lh[i]])), color = "red")+
    geom_vline(aes(xintercept = median(df[,lh[i]])),color='blue')+
    xlab(colnames(df)[lh[i]])+
    ylab("Frequency")+
    ggtitle(paste("Histogram, Median, and Mean of ",colnames(df)[lh[i]], ""))
  print(plot)
}
#   Now let's plot the boxplots to identify outliers from the variables
for(i in num_cols){
  print(         init+ 
                   geom_boxplot(aes(y=df[,colnames(df[,i,drop=F])]))+
                   ylab(colnames(df[,i,drop=F]))+
                   ggtitle(paste("Box plot for",colnames(df[,i,drop=F]))))
}

for(i in c(10,11,12)){
  print(         init+ 
                   geom_histogram(aes(x=df[,colnames(df[,i,drop=F])]),bins =50,color='black')+
                   xlab(colnames(df[,i,drop=F]))+
                   ggtitle(paste("Histogram plot for",colnames(df[,i,drop=F]))))
}


#install.packages('DMwR2')
library(DMwR2)
outlierImputer=function(df,num_cols){
  while (TRUE) {
    tot_miss=NULL
    for(i in num_cols){
      val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
      df[,i][df[,i] %in% val]= NA
      tot_miss=c(tot_miss,length(val))
    }
    print(sum(tot_miss))
    if(sum(tot_miss)>0){
      df=knnImputation(df,k = 3)
    }
    else{
      break
    }
  }
  return(df)
}
df=outlierImputer(df,num_cols)
#    Lets step into feature selection
#    Finding highly correlated numerical features with Count
# install.packages('corrgram')
library(corrgram)
numFeatureSel=function(df,num_cols){
  corr=cor(df[,num_cols])
  corrgram(df[,num_cols],order = F,upper.panel = panel.pie,text.panel = panel.txt,main='Correlation Plot')
}
numFeatureSel(df,num_cols)
# based on correlation plot we remove the highly correlated variables
df=subset(df,select = -c(Temp,Registered))
numFeatureSel(df,8:12)

#   

library(randomForest)
imppred=randomForest(Count ~ ., data = df,ntree = 100, keep.forest = FALSE, importance = TRUE)
importance(imppred, type = 1)
df=subset(df,select = -c(Holiday))

#     Feature Scaling
 #Standardisation
 #for(i in 7:10){
 #  df[,i] = (df[,i] - mean(df[,i]))/sd(df[,i])
 #}

df_scaled = scale(df[,7:10,drop=F])
df[,7:10]=df_scaled

#  Modeling the Data
# Divide data into train and test using stratified sampling method
library(caTools)
set.seed(101) 
sample = sample.split(df$Count, SplitRatio = .80)
train = subset(df, sample == TRUE)
test  = subset(df, sample == FALSE)

#Develop Model on training data

# Multiple Linear Regression Model
lr_model=lm(Count~.,train)
y_pred=predict(lr_model,test[,1:10])
#Summary of Linear model
summary(lr_model)
r=sum((y_pred-test[,11])^2)/sum((test[,11]-mean(test[,11]))^2)
1-r #-> 0.792

# Decision Tree Regression Model
# install.packages('rpart')
library(rpart)
dt_model=rpart(Count~.,train,method = 'anova')
y_pred_dt=predict(dt_model,test[,-11])
r=sum((y_pred_dt-test[,11])^2)/sum((test[,11]-mean(test[,11]))^2)
1-r #-> 0.792
summary(dt_model)
plot(dt_model, uniform = T, branch = 1, margin = 0.05, cex = 0.9)
text(dt_model, cex = 0.7)

# Random Forest Model
RF_model = randomForest(Count ~ ., train, importance = TRUE, ntree = 100)

library(inTrees)
treeList = RF2List(RF_model) 
exec = extractRules(treeList, train[,-11])
exec[1:6,]
readableRules = presentRules(exec, colnames(train))
readableRules[1:10,]
ruleMetric = getRuleMetric(exec, train[,-11], train$Count)  # get rule metrics
ruleMetric[1:2,]
y_pred_rf = predict(RF_model, test[,-11])
rt_sample=getTree(RF_model,k = 2,labelVar = T)

# KNN Regression Model
library(FNN)
knn_model=knn.reg(train = train[,7:10],test=test[,7:10],y=train[,11],k=3)
y_pred_knn=knn_model$pred

# Error Metrics
library(DMwR)
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}
print('Decision Tree Model')
MAPE(test[,11],y_pred_dt)
print('Linear Model')
MAPE(test[,11],y_pred)
print('Random Forest Model')
MAPE(test[,11],y_pred_rf)
print('Linear Model')
regr.eval(trues = test[,11],preds = y_pred,train.y = train[,11],stats = c('mae','mse','rmse','mape','nmse','nmae'))
print('Random Forest Model')
regr.eval(trues = test[,11],preds = y_pred_rf,train.y = train[,11],stats = c('mae','mse','rmse','mape','nmse','nmae'))
print('Decision Tree Model')
regr.eval(trues = test[,11],preds = y_pred_dt,train.y = train[,11],stats = c('mae','mse','rmse','mape','nmse','nmae'))
print('KNN Model')
regr.eval(trues = test[,11],preds = y_pred_knn,train.y = train[,11],stats = c('mae','mse','rmse','mape','nmse','nmae'))

predict_y=function(model,df){
  # Convert the Variables that are factors but were taken as numeric by R
  df$season=as.factor(df$season)
  df$yr=as.factor(df$yr)
  df$mnth=as.factor(df$mnth)
  df$holiday=as.factor(df$holiday)
  df$weekday=as.factor(df$weekday)
  df$workingday=as.factor(df$workingday)
  df$weathersit=as.factor(df$weathersit)
  df$dteday=as.character.Date(df$dteday)
  # Giving the columns meaningful Names 
  colnames(df)=c('Instant','Date','Season','Year','Month','Holiday','WeekDay','WorkingDay','WeatherSituation','Temp','ActualTemp','Humidity','WindSpeed','Casual','Registered')
  X=subset(df,select = -c(Instant,Date,Registered,Temp,Holiday))
  X[,7:10] = scale(X[,7:10], center=attr(df_scaled, "scaled:center"), 
                        scale=attr(df_scaled, "scaled:scale"))
  return(predict(model, X))
}

# Sample Input 
df_new=read.csv('./day.csv')
print('Sample Input')
print(df_new[1,-16])
print(paste0('Output for Sample Input with Linear Model',predict_y(lr_model,df_new[1,-16])))

