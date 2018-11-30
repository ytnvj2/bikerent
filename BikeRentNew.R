library(psych)
library(ggplot2)
library(DMwR2)
library(corrgram)
library(randomForest)
library(dummies)
library(car)
library(caTools)
library(rpart)
library(FNN)
library(DMwR)#regr. eval
set.seed(101) 
# Lets start with loading the data
df=read.csv('./day.csv')
# Lets see the structure of the data
str(df)
# Convert the Variables that are factors but were taken as numeric by R
df$season=as.factor(df$season)
#df$yr=as.factor(df$yr)
df$mnth=as.factor(df$mnth)
#df$holiday=as.factor(df$holiday)
df$weekday=as.factor(df$weekday)
#df$workingday=as.factor(df$workingday)
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
df$Casual=NULL
df$Registered=NULL
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

# Lets step into visualization
#  For Numerical Variables, we will be using the multi.hist fucntion to visualize
#  All variables in one go. The plots will contain each variable's histogram,
#  KDE plot, and a line representing Normal Distribution for comparison.
#   Import package from library

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
df=outlierImputer(df,num_cols[4:8])
#    Lets step into feature selection
#    Finding highly correlated numerical features with Count
# install.packages('corrgram')

# Feature Selection

numFeatureSel=function(df,num_cols){
  corr=cor(df[,num_cols])
  corrgram(df[,num_cols],order = F,upper.panel = panel.pie,text.panel = panel.txt,main='Correlation Plot')
}
numFeatureSel(df,num_cols)
# based on correlation plot we remove the highly correlated variable
df=subset(df,select = -c(Temp))
catFeatureSel=function(df,cat_cols){
  cSel=NULL
  for(i in cat_cols){
    x=summary(aov(Count~df[,i], data = df))
    print(x)
    if(x[[1]]$`Pr(>F)`[1]<0.05){
      cSel=c(cSel,i)
    }
  }
  return(cSel)
}
cSel=catFeatureSel(df,cat_cols)

# Remove Weekday as not significant
df=subset(df,select = -c(WeekDay))

# Convert categorical variables to dummy variables
df=dummy.data.frame(df, sep = "." )
df=df[,-c(1,6,20)]

# Calculate VIF for all the variables and select variables with low VIF
VIF_check=function(df_o){
  while (T) {
    lr_model=lm(Count~.,data = df_o)
    x=vif(lr_model)
    maxVar = max(x)
    if (maxVar > 6){
      j = which(x == maxVar)
      df_o = df_o[, -j]
    }
    else{
      break()
    }
  }
  return(df_o)
}
df_vif=VIF_check(df)

# Conduct BackElim RandomForest to find important variables
backwardEliminationRF=function(df,sl){
  numVars = length(df)
  for (i in c(1:numVars)){
    imppred=randomForest(formula =Count ~ ., data = df,ntree = 100, keep.forest = FALSE, importance = TRUE)
    minVar =min(importance(imppred, type = 1))
    if (minVar < sl){
      j = which(importance(imppred, type = 1) == minVar)
      df = df[, -j]
    }
    numVars = numVars - 1
  }
  return(imppred)
  
}

rf_model=backwardEliminationRF(df_vif,5)
imp=importance(rf_model)

# Select only most important variables
df_vif=df_vif[,c(row.names(imp),'Count')]

df_copy=df

# Conduct Linear BackElim to get most significant variables for linear model
backwardElimination=function(df,sl){
  numVars = length(df)
  for (i in c(1:numVars)){
    regressor = lm(formula = Count~., data = df)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      df = df[, -j]
    }
    numVars = numVars - 1
  }
  return(regressor)
  
}
lr_model=backwardElimination(df_copy,0.01)
summary(lr_model)
df_lr=df[,c(names(lr_model$coefficients)[-1],'Count')]

# feature scaling
df_scaled = scale(df_vif[,-19])
df_vif[,-19]=df_scaled


#  Modeling the Data
# Divide data into train and test using stratified sampling method

# For Linear BackElim Variables
sample = sample.split(df_lr$Count, SplitRatio = 0.8)
train = subset(df_lr, sample == TRUE)
test  = subset(df_lr, sample == FALSE)


#Develop Model on training data

# Multiple Linear Regression Model
lr_model=backwardElimination(train,0.05)
summary(lr_model)
lr=lm(Count~.,train)
y_pred=predict(lr_model,test[,1:14])
#Summary of Linear model
summary(lr_model)
# Visualizing the linear model
plot(lr_model) 
r=sum((y_pred-test[,15])^2)/sum((test[,15]-mean(test[,15]))^2)
1-r #-> 0.819

# Decision Tree Regression Model
# install.packages('rpart')

# For VIF and RF Variables
sample = sample.split(df_vif$Count, SplitRatio = 0.8)
train = subset(df_vif, sample == TRUE)
test  = subset(df_vif, sample == FALSE)

dt_model=rpart(Count~.,train,method = 'anova')
y_pred_dt=predict(dt_model,test[,-19])
r=sum((y_pred_dt-test[,19])^2)/sum((test[,19]-mean(test[,19]))^2)
1-r #-> 0.657
summary(dt_model)
plot(dt_model, uniform = T, branch = 1, margin = 0.05, cex = 0.9)
text(dt_model, cex = 0.7)

# Random Forest Model
RF_model = randomForest(Count ~ ., train, importance = TRUE, ntree = 100)
summary(RF_model)
y_pred_rf = predict(RF_model, test[,-19])
r=sum((y_pred_rf-test[,19])^2)/sum((test[,19]-mean(test[,19]))^2)
1-r #-> 0.797
plot(RF_model)

# KNN Regression Model
knn_model=knn.reg(train = train[,-19],test=test[,-19],y=train[,19],k=7)
y_pred_knn=knn_model$pred
r=sum((y_pred_knn-test[,19])^2)/sum((test[,19]-mean(test[,19]))^2)
1-r #-> 0.832

# Error Metrics

#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

MAPE(test[,19],y_pred_dt)
MAPE(test[,19],y_pred)
MAPE(test[,19],y_pred_rf)

regr.eval(trues = test[,19],preds = y_pred,train.y = train[,19],stats = c('mae','mse','rmse','mape','nmse','nmae'))
regr.eval(trues = test[,19],preds = y_pred_rf,train.y = train[,19],stats = c('mae','mse','rmse','mape','nmse','nmae'))
regr.eval(trues = test[,19],preds = y_pred_dt,train.y = train[,19],stats = c('mae','mse','rmse','mape','nmse','nmae'))
regr.eval(trues = test[,19],preds = y_pred_knn,train.y = train[,19],stats = c('mae','mse','rmse','mape','nmse','nmae'))

predict_y=function(model,df,type){
  # Convert the Variables that are factors but were taken as numeric by R
  df$season=as.factor(df$season)
  df$mnth=as.factor(df$mnth)
  df$weekday=as.factor(df$weekday)
  df$weathersit=as.factor(df$weathersit)
  df$dteday=as.character.Date(df$dteday)
  # Giving the columns meaningful Names 
  colnames(df)=c('Instant','Date','Season','Year','Month','Holiday','WeekDay','WorkingDay','WeatherSituation','Temp','ActualTemp','Humidity','WindSpeed','Casual','Registered')
  print('Sample Input')
  print(df[3,])
  X=subset(df,select = -c(Instant,Date,Registered,Casual,Temp,WeekDay))
  X=dummy.data.frame(X,sep='.')
  X=X[,-c(1,6,20)]
  #X[,] = scale(X, center=attr(df_scaled, "scaled:center"),scale=attr(df_scaled, "scaled:scale"))
  if(type=='linear'){
    X=X[,-c(5,7:10,14,15,17)]
  }
  else{
    X=X[,-c(2,16,17,20)]
  }
  return(predict(model, X[3,]))
}


# Sample Input 
df_new=read.csv('day.csv')
print(paste0('Output for Sample Input with Linear Model',predict_y(lr_model,df_new[,-16],'linear')))
print(paste0('Output for Sample Input with Decision Tree Model',predict_y(dt_model,df_new[,-16],'DT')))
print(paste0('Output for Sample Input with Random Forest Model',predict_y(RF_model,df_new[,-16],'RF')))

