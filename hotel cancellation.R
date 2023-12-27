library(ggplot2)
library(dplyr)
df=read.csv("C:/Users/82102/Downloads/archive/hotel_bookings.csv")
head(df)

#데이터전처리
print(colSums(is.na(df) | sapply(df, is.null)))
df <- df[, !(names(df) %in% c('agent', 'company'))]
df <- na.omit(df)
df <- df[!(df$children == 0 & df$adults == 0 & df$babies == 0), ]
df <- df[!(df$adr < 0), , drop = FALSE]


#hoteltype
hotel_type<-table(data$hotel)
pie_chart <- ggplot(data.frame(hotel_type), aes(x = "", y = Freq, fill = Var1)) +
  geom_bar(stat = "identity", width = 1, color = "white") +
  coord_polar("y") +
  theme_void() +
  ggtitle("Hotel Types")
pie_chart + geom_text(aes(label = scales::percent(as.numeric(hotel_type) / sum(hotel_type))), position = position_stack(vjust = 0.5))

#hoteltype에 따른 price
library(plotly)
data <- df[df$is_canceled == 0, ]
plot <- plot_ly(data, x = ~reserved_room_type, y = ~adr, color = ~hotel, type = "box") %>%
  layout(title = "Box Plot of ADR by Reserved Room Type and Hotel",
         xaxis = list(title = "Reserved Room Type"),
         yaxis = list(title = "ADR"))
print(plot)

#rate of cacellation
cancellation_counts <- table(df$is_canceled)
ggplot(df, aes(x = factor(is_canceled), fill = factor(is_canceled))) +
  geom_bar() +
  scale_x_discrete(labels = c('Not Canceled', 'Canceled')) +
  labs(title = "Cancellation Counts", x = "Cancellation Status", y = "Count") +
  theme_minimal()

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

#predict cancellation
df$is_canceled <- as.numeric(df$is_canceled)
numeric_columns <- sapply(df, is.numeric)
numeric_data <- df[, numeric_columns]
cancel_corr <- cor(df$is_canceled, numeric_data)
print(cancel_corr)
#reservation_status 확인
result <- df %>%
  group_by(is_canceled, reservation_status) %>%
  summarise(count = n())
print(result)


library(caret)
library(mlr)
num_features <- c("lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                  "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
                  "babies", "is_repeated_guest", "previous_cancellations",
                  "previous_bookings_not_canceled","adr",
                  "required_car_parking_spaces", "total_of_special_requests")

cat_features <- c("hotel", "arrival_date_month", "meal", "market_segment",
                  "distribution_channel", "reserved_room_type", "deposit_type", "customer_type")

features <- c(num_features, cat_features)

# Separate features and predicted value
X <- df %>%
  select(-is_canceled, one_of(features))
y <- df$is_canceled

# Preprocessing for numerical features
num_transformer <- preProcess(X[, num_features])
# Preprocessing for categorical features
dummy_matrix <- preProcess(X[, cat_features])
preprocessor <- preProcess(list(num_transformer, cat_transformer), df[, c(num_features, cat_features)])



#PreProcessing
library(corrplot)
# 상관 행렬 계산
numeric_columns <- sapply(df, is.numeric)
numeric_data <- df[, numeric_columns]
cor_matrix <- cor(numeric_data)
# 히트맵 그리기
corrplot(cor_matrix, method = "color", addCoef.col = "black", order = "hclust", type = "upper", tl.col = "black", tl.srt = 45)

# 삭제할 열의 목록
useless_col <- c('days_in_waiting_list', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
                 'reservation_status', 'country', 'days_in_waiting_list')
# 열 삭제
df <- df %>%
  select(-one_of(useless_col))
names(df)

cat_cols <- df %>%
  select_if(is.character) %>%
  names()
cat_df = df[, cat_cols]
head(cat_df)

# 문자열 열 제외한 열 선택
num_df <- df %>%
  select(-one_of(names(df)[sapply(df, is.character)])) %>%
  select(-is_canceled)
apply(num_df, 2, var)
df$adr <- ifelse(is.na(df$adr), mean(df$adr, na.rm = TRUE), df$adr)
num_df <- num_df %>%
  mutate(
    lead_time = log(lead_time + 1),
    arrival_date_week_number = log(arrival_date_week_number + 1),
    arrival_date_day_of_month = log(arrival_date_day_of_month + 1),
    adr = log(adr+1)
)
num_df$adr <- ifelse(is.na(num_df$adr), mean(num_df$adr, na.rm = TRUE), num_df$adr)
apply(num_df, 2, var)

X <- num_df
y <- df[,'is_canceled', drop=FALSE]


#Logistic Regression
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

#Random Forest 
set.seed(1)
rf_training_model<-randomForest(is_canceled~.,    # model formula
                                data=training1,          # use a training dataset for building a model
                                ntree=500,                     
                                cutoff=c(0.5,0.5), 
                                mtry=2,
                                importance=TRUE)
set.seed(1)
res<-tuneRF(x = training_1%>%select(-is_canceled),
            y = training_1$is_canceled,mtryStart=2,
            ntreeTry = 500)
