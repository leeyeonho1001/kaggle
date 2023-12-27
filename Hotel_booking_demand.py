{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP10maWGEBub/qkk0B8EPf+",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/leeyeonho1001/kaggle/blob/main/Hotel_booking_demand.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#File uploading\n",
        "import pandas as pd\n",
        "df = pd.read_csv('hotel_bookings.csv')\n",
        "print(df.head())"
      ],
      "metadata": {
        "id": "2mg0FIV9ZyS5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Hotel type\n",
        "import matplotlib.pyplot as plt\n",
        "hotel_type = df['hotel'].value_counts()\n",
        "#plot\n",
        "plt.figure(figsize=(8, 8))\n",
        "plt.pie(hotel_type, labels=hotel_type.index, autopct=lambda p: '{:.1f}%'.format(p))\n",
        "plt.title(\"Hotel Types\")\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "KUjtadjbaDvV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Rate of cancellation\n",
        "import seaborn as sns\n",
        "cancellation_counts = df['is_canceled'].value_counts()\n",
        "#plot\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.barplot(x=cancellation_counts.index, y=cancellation_counts.values, palette=\"viridis\")\n",
        "plt.title(\"Cancellation Counts\")\n",
        "plt.xlabel(\"Cancellation Status\")\n",
        "plt.ylabel(\"Count\")\n",
        "plt.xticks(ticks=[0, 1], labels=['Not Canceled', 'Canceled'])\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "qpvbFaGXaOdD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Monthly cancellation rate\n",
        "ordered_months = [\"January\", \"February\", \"March\", \"April\", \"May\", \"June\",\n",
        "                  \"July\", \"August\", \"September\", \"October\", \"November\", \"December\"]\n",
        "res_book_per_month = df[df['hotel'] == \"Resort Hotel\"].groupby('arrival_date_month').agg(\n",
        "    Bookings=pd.NamedAgg(column='is_canceled', aggfunc='count'),\n",
        "    Cancellations=pd.NamedAgg(column='is_canceled', aggfunc='sum')\n",
        ").reset_index()\n",
        "res_book_per_month['Hotel'] = \"Resort Hotel\"\n",
        "cty_book_per_month = df[df['hotel'] == \"City Hotel\"].groupby('arrival_date_month').agg(\n",
        "    Bookings=pd.NamedAgg(column='is_canceled', aggfunc='count'),\n",
        "    Cancellations=pd.NamedAgg(column='is_canceled', aggfunc='sum')\n",
        ").reset_index()\n",
        "cty_book_per_month['Hotel'] = \"City Hotel\"\n",
        "full_cancel_data = pd.concat([res_book_per_month, cty_book_per_month])\n",
        "full_cancel_data['cancel_percent'] = (full_cancel_data['Cancellations'] / full_cancel_data['Bookings']) * 100\n",
        "#plot\n",
        "plt.figure(figsize=(12, 6))\n",
        "sns.barplot(x='arrival_date_month', y='cancel_percent', hue='Hotel', data=full_cancel_data,\n",
        "            palette={\"City Hotel\": \"blue\", \"Resort Hotel\": \"orange\"})\n",
        "plt.title(\"Cancellations per month\")\n",
        "plt.xlabel(\"Month\")\n",
        "plt.ylabel(\"Cancellations [%]\")\n",
        "plt.xticks(rotation=45, ha='right')\n",
        "plt.legend(title=\"Hotel\", loc=\"upper right\")\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "aszJnfl_anO1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Type of hotel canceled\n",
        "canceled_hoteltype = df[['is_canceled', 'hotel']]\n",
        "canceled_hotel = canceled_hoteltype[canceled_hoteltype['is_canceled'] == 1].groupby('hotel').size().reset_index(name='count')\n",
        "#plot\n",
        "plt.figure(figsize=(8, 6))\n",
        "sns.barplot(x='hotel', y='count', data=canceled_hotel, palette='viridis')\n",
        "plt.title('Cancellation rates by hotel type')\n",
        "plt.xlabel('Hotel Type')\n",
        "plt.ylabel('Count')\n",
        "for index, value in enumerate(canceled_hotel['count']):\n",
        "    plt.text(index, value, str(value), ha='center', va='bottom', fontsize=9)\n",
        "plt.show()"
      ],
      "metadata": {
        "id": "_Vo-G_CVbdxJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Preprocessing\n",
        "df = df.drop(['agent', 'company'], axis=1)\n",
        "df = df.dropna()\n",
        "df = df[~((df['children'] == 0) & (df['adults'] == 0) & (df['babies'] == 0))]\n",
        "df = df[df['adr'] >= 0]\n",
        "df = df.select_dtypes(include='number')\n",
        "cor_matrix = df.corr()\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.heatmap(cor_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)\n",
        "plt.title('Correlation Matrix')\n",
        "plt.show()\n",
        "useless_col = ['days_in_waiting_list', 'arrival_date_year', 'booking_changes']\n",
        "df = df.drop(columns=useless_col)"
      ],
      "metadata": {
        "id": "-qc8b8pjcMIY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(df.head())"
      ],
      "metadata": {
        "id": "BK5jGFpfjKSB"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Logistic Regression model\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
        "from sklearn.model_selection import train_test_split\n",
        "import pandas as pd\n",
        "\n",
        "X = df.drop(columns=['is_canceled'])\n",
        "y = df['is_canceled']\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)\n",
        "\n",
        "#Fit model\n",
        "lr = LogisticRegression(random_state=1)\n",
        "lr.fit(X_train, y_train)\n",
        "y_pred_lr = lr.predict(X_test)\n",
        "#Evaluate model\n",
        "acc_lr = accuracy_score(y_test, y_pred_lr)\n",
        "conf = confusion_matrix(y_test, y_pred_lr)\n",
        "clf_report = classification_report(y_test, y_pred_lr)\n",
        "print(f\"Accuracy Score : {acc_lr}\")\n",
        "print(f\"Confusion Matrix:\\n{conf}\")\n",
        "print(f\"Classification Report:\\n{clf_report}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xE6SP-OCenlj",
        "outputId": "03195a81-b152-46be-b8e7-e85f4fb6dffa"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy Score of Logistic Regression is: 0.6960648148148149\n",
            "Confusion Matrix:\n",
            "[[4425  879]\n",
            " [1747 1589]]\n",
            "Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.72      0.83      0.77      5304\n",
            "           1       0.64      0.48      0.55      3336\n",
            "\n",
            "    accuracy                           0.70      8640\n",
            "   macro avg       0.68      0.66      0.66      8640\n",
            "weighted avg       0.69      0.70      0.68      8640\n",
            "\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Decision Tree classifier\n",
        "import pandas as pd\n",
        "from sklearn.tree import DecisionTreeClassifier, export_text\n",
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn.metrics import confusion_matrix, accuracy_score\n",
        "\n",
        "#Fit model\n",
        "dtc = DecisionTreeClassifier(random_state=1)\n",
        "dtc.fit(X_train, y_train)\n",
        "y_pred_dtc = dtc.predict(X_test)\n",
        "#Evaluate model\n",
        "acc_dtc = accuracy_score(y_test, y_pred_dtc)\n",
        "conf = confusion_matrix(y_test, y_pred_dtc)\n",
        "clf_report = classification_report(y_test, y_pred_dtc)\n",
        "print(f\"Accuracy Score : {acc_dtc}\")\n",
        "print(f\"Confusion Matrix : \\n{conf}\")\n",
        "print(f\"Classification Report : \\n{clf_report}\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y23P89e5fycA",
        "outputId": "cca0d47b-6f0b-446d-9476-e5c39228363f"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy Score of Decision Tree is : 0.7923611111111111\n",
            "Confusion Matrix : \n",
            "[[4403  901]\n",
            " [ 893 2443]]\n",
            "Classification Report : \n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.83      0.83      0.83      5304\n",
            "           1       0.73      0.73      0.73      3336\n",
            "\n",
            "    accuracy                           0.79      8640\n",
            "   macro avg       0.78      0.78      0.78      8640\n",
            "weighted avg       0.79      0.79      0.79      8640\n",
            "\n"
          ]
        }
      ]
    }
  ]
}