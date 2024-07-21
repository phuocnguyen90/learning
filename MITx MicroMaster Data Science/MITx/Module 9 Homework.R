
# Q1-2 difference between fast-food restaurants located in NJ and Pennsylvania
model <- lm(empft ~ state, data = fast_food_data)
summary(model)

# Q3-5 average wage in Pennsylvania prior to the change?
model2<-lm(formula = wage_st ~ state, data = fast_food_data)
summary(model2)
 
# Q6: difference in full time employment between the restaurants located in the 
# northeast suburbs of Philadelphia and the rest of Pennsylvania
model3 <- lm(empft ~ state + pa1 + pa2, data = fast_food_data)
summary(model3)

# Q7: using dummy variable to indicate the change
fast_food_data$post <- ifelse(fast_food_data$empft2 - fast_food_data$empft > 0, 1, 0)
model4 <- lm(empft ~ state + post + state:post, data = fast_food_data)
summary(model4)

# Q8: alternative model for Q7
fast_food_data$empft_diff <- fast_food_data$empft2 - fast_food_data$empft
model5 <- lm(empft_diff ~ state, data = fast_food_data)
summary(model5)