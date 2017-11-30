#reading data
eye.dia<-read.csv(file.choose())
eye.scale <- scale(eye.dia)
#splitting data
smp_size <- floor(0.60 * nrow(eye.scale))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(eye.scale)), size = smp_size)
train <- eye.scale[train_ind, ]
test <- eye.scale[-train_ind, ]



#Creating model
eye.model <-  glm(Class~., data = train, family = binomial)
#Testing with test data
predict.model <- predict(eye.model, newdata = test, type = "response")
#Calculating error
mean(predict.model != test$Class)

#Typing new data to predict
new.data.2 <- data.frame(1, 22, 22 , 22, 19 , 18, 14,49.89, 17.77, 5.27, 0.77, 0.0186, 0.0068, 0.0039, 0.0039, 0.4869, 0.100, 1)
column.names <- c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R")
names(new.data.2) <- column.names

#Predicting values
predict.newdata <- predict(eye.model, new.data.2, type = "response")
#Finding the probability
predict.newdata

## build the neural network (NN)
#Train the neural network
#Going to have 7 hidden layers
#Threshold is a numeric value specifying the threshold for the partial
#derivatives of the error function as stopping criteria.
install.packages("neuralnet")
library(neuralnet)
eye.net <- neuralnet(Class~A+B+C+D+E+F+G+H+I+J+K+L+M+N+O+P+Q+R, data = train, hidden = 7,lifesign = "minimal",linear.output = T, threshold=0.1)

#Plot the neural network
plot(eye.net, rep = "best")

## test the resulting output
temp_test <- subset(test, select = c("A", "B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R"))
eye.net.results <- compute(eye.net, temp_test)
head(temp_test)


head(test)
results <- data.frame(actual = test[,19], prediction =eye.net.results$net.result)
results[100:115, ]
results$prediction <- round(results$prediction)
results[100:115, ]

##Tree Classification

install.packages("rpart")






