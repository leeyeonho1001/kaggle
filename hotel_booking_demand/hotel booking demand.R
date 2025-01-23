library(ggplot2)
library(dplyr)
df=read.csv("C:/Users/82102/Downloads/archive/hotel_bookings.csv")
head(df)
df <- df[, !(names(df) %in% c('agent', 'company'))]
df <- na.omit(df)
df <- df[!(df$children == 0 & df$adults == 0 & df$babies == 0), ]
df <- df[!(df$adr < 0), , drop = FALSE]

#EDA
#hoteltype
hotel_type<-table(data$hotel)
pie_chart <- ggplot(data.frame(hotel_type), aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y") +
  theme_void() +
  ggtitle("Hotel Types")
pie_chart + geom_text(aes(label = scales::percent(as.numeric(hotel_type) / sum(hotel_type))), position = position_stack(vjust = 0.5))
#rate of cancellation
cancellation_counts <- table(df$is_canceled)
ggplot(df, aes(x = factor(is_canceled), fill = factor(is_canceled))) +
  geom_bar() +
  scale_x_discrete(labels = c('Not Canceled', 'Canceled')) +
  labs(title = "Cancellation Counts", x = "Cancellation Status", y = "Count") +
  theme_minimal()
#Room price per night over the months
data_resort <- df %>%
  filter(hotel == 'Resort Hotel', is_canceled == 0) %>%
  group_by(arrival_date_month) %>%
  summarise(price_for_resort = mean(adr))
data_city <- df %>%
  filter(hotel == 'City Hotel', is_canceled == 0) %>%
  group_by(arrival_date_month) %>%
  summarise(price_for_city_hotel = mean(adr))
final_hotel <- merge(data_resort, data_city, by = 'arrival_date_month')
names(final_hotel) <- c('month', 'price_for_resort', 'price_for_city_hotel')
custom_month_order <- c("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
final_prices <- final_hotel[order(match(final_hotel$month, custom_month_order)), ]
#시각화
final_prices$month <- factor(final_prices$month, levels = custom_month_order)
ggplot(final_prices, aes(x = month)) +
  geom_line(aes(y = price_for_resort, group=1, color = "Resort Hotel"), size=1.5) +
  geom_line(aes(y = price_for_city_hotel, group=1, color = "City Hotel"), size=1.5) +
  labs(title = "Room price per night over the Months",
       x = "Month",
       y = "Price",
       color = "Hotel Type") +
  theme_dark() +
  scale_color_manual(values = c("Resort Hotel" = "blue", "City Hotel" = "red")) +
  theme_minimal()
#월별 cancellation rate
res_book_per_month <- df %>%
  filter(hotel == "Resort Hotel") %>%
  group_by(arrival_date_month) %>%
  summarize(Bookings = n(), Cancellations = sum(is_canceled)) %>%
  mutate(Hotel = "Resort Hotel")
cty_book_per_month <- df %>%
  filter(hotel == "City Hotel") %>%
  group_by(arrival_date_month) %>%
  summarize(Bookings = n(), Cancellations = sum(is_canceled)) %>%
  mutate(Hotel = "City Hotel")
full_cancel_data <- bind_rows(res_book_per_month, cty_book_per_month) %>%
  mutate(cancel_percent = Cancellations / Bookings * 100)
ordered_months <- c("January", "February", "March", "April", "May", "June", 
                    "July", "August", "September", "October", "November", "December")
full_cancel_data$arrival_date_month <- factor(full_cancel_data$arrival_date_month, 
                                              levels = ordered_months, ordered = TRUE)
ggplot(full_cancel_data, aes(x = arrival_date_month, y = cancel_percent, fill = Hotel)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Cancellations per month",
       x = "Month",
       y = "Cancellations [%]") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c("City Hotel" = "blue", "Resort Hotel" = "orange")) +
  guides(fill = guide_legend(title = "Hotel")) +
  theme(legend.position = "upper right")
#type of the hotel canceled
canceled_hoteltype <- df %>% select(is_canceled, hotel)
canceled_hotel <- canceled_hoteltype %>%
  filter(is_canceled == 1) %>%
  group_by(hotel) %>%
  summarise(count = n())
ggplot(canceled_hotel, aes(x = hotel, y = count, fill = hotel)) +
  geom_bar(stat = "identity") +
  labs(title = 'Cancellation rates by hotel type', x = 'Hotel Type', y = 'Count') +
  geom_text(aes(label = sprintf("%d", count)), vjust = -0.5, size = 3) +
  theme_minimal()

#PreProcessing
df <- df[, !(names(df) %in% c('agent', 'company'))]
df <- na.omit(df)
df <- df[!(df$children == 0 & df$adults == 0 & df$babies == 0), ]
df <- df[!(df$adr < 0), , drop = FALSE]
library(corrplot)
numeric_columns <- sapply(df, is.numeric)
numeric_data <- df[, numeric_columns]
cor_matrix <- cor(numeric_data)
corrplot(cor_matrix, method = "color", type = "upper")
useless_col <- c('days_in_waiting_list', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
                 'reservation_status', 'country')
df <- df %>%
  select(-one_of(useless_col))
names(df)

#Modeling
#Logistic Regression
library(glm)
set.seed(1)
index=sample(nrow(df), nrow(df)*0.3)
df<- df %>%
  filter(market_segment!='Undefined')
test=df[index, ]
training=df[-index, ]
training1=training[c('hotel','is_canceled','lead_time','adults','children','babies','meal',
                     'market_segment','distribution_channel','is_repeated_guest',
                     'previous_cancellations','previous_bookings_not_canceled','reserved_room_type',
                     'deposit_type','customer_type','adr',
                     'required_car_parking_spaces')]
lr_model=glm(is_canceled~., family='binomial',data=training1)
test$lr_pred_prob<-predict(lr_model, test, type='response')
test$lr_pred_class<-ifelse(test$lr_pred_prob >0.5, '1', '0')
table(test$is_canceled == test$lr_pred_class)

table(test$lr_pred_class, test$is_canceled,dnn=c('predicted', 'actual'))
27829/nrow(test)

#Classification Tree
library(rpart)
library(rpart.plot)
ct_model<-rpart(is_canceled~., data=training1, method='class', control=rpart.control(cp=0.03))
rpart.plot(ct_model)
test$ct_pred_prob<-predict(ct_model, test)[,2]
test$ct_pred_class<-predict(ct_model, test, type='class')
table(test$is_canceled==test$ct_pred_class)
#K-cross validation
set.seed(1)
full_tree<-rpart(is_canceled~.,data=training1,method='class',control=rpart.control(cp=0, maxdepth=3)) 
rpart.plot(full_tree)
printcp(full_tree)
plotcp(full_tree)
min_xerror<-full_tree$cptable[which.min(full_tree$cptable[,"xerror"]),]
min_xerror
min_xerror_tree<-prune(full_tree, cp=min_xerror[1])
rpart.plot(min_xerror_tree)
bp_tree<-min_xerror_tree
test$ct_bp_pred_prob<-predict(bp_tree,test)[,2]
test$ct_bp_pred_class=ifelse(test$ct_bp_pred_prob>0.5,"Yes","No")

table(test$ct_bp_pred_class,test$is_canceled, dnn=c("predicted","actual"))
(22417+4980)/nrow(test)
