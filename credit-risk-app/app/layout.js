import "./globals.css";

export const metadata = {
  title: "Credit Risk Analysis — ML Ensemble Predictor",
  description:
    "Evaluate credit default risk using 15 machine learning models. Input customer data and receive comprehensive risk scores with feature importance analysis.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
