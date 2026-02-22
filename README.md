### CARP

**CARP** is a hospital artificial intelligence infrastructure which leverages differential privacy to improve a public AI model without hospitals revealing sensitive patient information. The project was created by Peter Liu, Charles Liu, Alex Duong, and Russell Habib for the 2026 Crimson Code Hackathon (Advanced Track).

The project consists of two main components:
1. Hospital Client (local)
2. Main AI Server (global)

The _hospital client_ allows hospitals to upload sensitive information (through CSVs) locally and train AI models with differential privacy to collect weights. Hospitals can test out the accuracy of their model through this local website. Hospitals can customize $\epsilon$, the privacy/accuracy tradeoff to decide how much they want to contribute to the global AI model on the public server. $\epsilon = 1$ is a standard reasonable amount to satisfy HIPPA. When ready, hospitals can upload their weights to the main AI server to try and improve the public model. This portion is _dockerized_ to ensure no environment issues and smooth usage in hospitals, saving time on debugging and package management.

The _main AI server_ is a public website which allows hospitals to upload weights to improve a public AI which anyone can query. For this prototype, we focus on analyzing diabetes risk. Users can enter their personal information and get a percentage analyzing their risk of diabetes.

### Impact

This project isn't just a diabetes predictor, which is a classical machine learning exercise you can find anywhere. Instead of merely reinventing the wheel, we are proving a concept and realizing an opportunity. Many fields like finance and healthcare have sensitive data which understandably they don't want to reveal. However, differential privacy offers a means to leverage this data while respecting privacy concerns. This data could be used to train AI models which can not just improve revenue, delight customers, and advance research---but save lives. Our project is more than a reinventing the wheel, its a statement: privacy-preserving computation is an underlooked tool which people can leverage for greater good.

