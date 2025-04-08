import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def generate_forecast_exception_report(forecast_df, error_threshold=20.0):
    """
    Generate a forecast exception report based on absolute percentage error.

    Args:
        forecast_df (pd.DataFrame): DataFrame with columns ['date', 'forecast', 'actual']
        error_threshold (float): Threshold in percentage to flag exceptions

    Returns:
        pd.DataFrame: Exception report
    """

    # Calculate Absolute Error and Error %
    forecast_df['absolute_error'] = (forecast_df['forecast'] - forecast_df['actual']).abs()
    forecast_df['error_percent'] = (forecast_df['absolute_error'] / forecast_df['actual'].replace(0, 1)) * 100  # avoid div by 0

    # Flag exceptions
    forecast_df['is_exception'] = forecast_df['error_percent'] > error_threshold

    # Select only exceptions
    exception_report = forecast_df[forecast_df['is_exception']].copy()

    return exception_report


def send_email(subject, body, to_emails, smtp_server, smtp_port, smtp_user, smtp_password):
    """
    Send an email with the given subject and body.

    Args:
        subject (str): Email subject
        body (str): Email body
        to_emails (list): List of recipient emails
        smtp_server (str): SMTP server address
        smtp_port (int): SMTP server port
        smtp_user (str): SMTP username
        smtp_password (str): SMTP password
    """

    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = ", ".join(to_emails)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, to_emails, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Example Usage
if __name__ == "__main__":
    # Sample data
    data = {
        'date': ['2025-04-01', '2025-04-02', '2025-04-03'],
        'forecast': [500, 520, 530],
        'actual': [700, 515, 800]
    }
    forecast_df = pd.DataFrame(data)

    exception_report = generate_forecast_exception_report(forecast_df, error_threshold=20.0)
    
    if not exception_report.empty:
        print("=== Forecast Exception Report ===")
        print(exception_report)

        # Save report locally
        report_path = 'forecast_exception_report.csv'
        exception_report.to_csv(report_path, index=False)

        # Email setup
        subject = "⚠️ Forecast Exception Alert"
        body = f"Forecast exceptions detected! Please find the report attached or check system logs.\n\nSummary:\n{exception_report[['date', 'forecast', 'actual', 'error_percent']]}"
        to_emails = ["recipient@example.com"]  # Replace with actual emails
        smtp_server = "smtp.gmail.com"         # Replace with your SMTP server
        smtp_port = 587
        smtp_user = "your_email@example.com"   # Replace with your email
        smtp_password = "your_password"        # Replace with your password

        send_email(subject, body, to_emails, smtp_server, smtp_port, smtp_user, smtp_password)

    else:
        print("No forecast exceptions found.")
