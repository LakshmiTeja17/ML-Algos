
library(tictoc)
my_data <- read.table("data.txt",sep="\t",header=FALSE)

x1 <- unlist(my_data[1], use.names = FALSE)
x2 <- unlist(my_data[2], use.names = FALSE)
x3 <- unlist(my_data[3], use.names = FALSE)
x4 <- unlist(my_data[4], use.names = FALSE)
y <- unlist(my_data[5], use.names = FALSE)

model <- lm(y ~ x1 + x2 + x3 + x4)

y_pred <- predict(model)
y_pred

plot(y,y_pred, col="blue", pch=20)

sum((y-y_pred)^2)/length(y)

gradientDesc <- function(x1,x2,x3,x4,y, learn_rate, n, max_iter) {
  m1 <- runif(1, 0, 1)
  m2 <- runif(1, 0, 1)
  m3 <- runif(1, 0, 1)
  m4 <- runif(1, 0, 1)
  c <- runif(1, 0, 1)
  ynew <- m1*x1 + m2*x2 + m3*x3 + m4*x4 + c
  MSE <- sum((y - ynew) ^ 2) / n
  converged = F
  iterations = 0
  while(converged == F) {
    ## Implement the gradient descent algorithm
    m1_new <- m1 - learn_rate * ((1 / n) * (sum((ynew - y) * x1)))
    m2_new <- m2 - learn_rate * ((1 / n) * (sum((ynew - y) * x2)))
    m3_new <- m3 - learn_rate * ((1 / n) * (sum((ynew - y) * x3)))
    m4_new <- m4 - learn_rate * ((1 / n) * (sum((ynew - y) * x4)))
    c_new <- c - learn_rate * ((1 / n) * (sum(ynew - y)))
    m1 <- m1_new
    m2 <- m2_new
    m3 <- m3_new
    m4 <- m4_new
    c <- c_new
    ynew <- m1*x1 + m2*x2 + m3*x3 + m4*x4 + c
    iterations = iterations + 1
    if(iterations > max_iter) { 
      converged = T
      return(paste("MSE:", sum((y-ynew)^2)/n))
    }
  }
}

start_time <- Sys.time()
gradientDesc(x1,x2,x3,x4,y,0.0001,46,10000000)
end_time <- Sys.time()
end_time - start_time



