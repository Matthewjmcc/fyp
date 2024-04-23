library(ggplot2)
library(lsr)
setwd('C:\\Users\\matth\\OneDrive\\Desktop\\Fourth_Year\\Final Year Project\\fyp\\data\\evaluation_results\\cnn')
getwd()

el_gt <- read.csv('el_gt.csv')
th_gt <- read.csv('th_gt.csv')
th_vae <- read.csv('vae_th_b20l3.csv')
th_vae2 <- read.csv('vae_th_b8l15.csv')
el_vae <- read.csv('vae_el_b24l3.csv')
el_vae2 <- read.csv('vae_el_b8l10.csv')
th_gan <- read.csv('gan_th_b8e1000.csv')
th_gan2 <- read.csv('gan_th_b16e2000.csv')
el_gan <- read.csv('gan_el_b20e500.csv')
el_gan2 <- read.csv('gan_el_b24e1000.csv')

th_b_vae <- read.csv('vae_blended_th_b20l3.csv')
th_b_vae2 <- read.csv('vae_blended_th_b24l15.csv')
el_b_vae <- read.csv('vae_blended_el_b24l3.csv')
el_b_vae2 <- read.csv('vae_blended_el_b20l3.csv')
th_b_gan <- read.csv('gan_blended_th_b6e100.csv')
th_b_gan2 <- read.csv('gan_blended_th_b20e100.csv')
el_b_gan <- read.csv('gan_blended_el_b20e500.csv')
el_b_gan2 <- read.csv('gan_blended_el_b10e100.csv')


th_mse <- cbind(th_gt$mse, th_b_vae$mse, th_b_gan$mse)
el_mse <- cbind(el_gt$mse, el_b_vae$mse, el_b_gan$mse)
df_th <- data.frame(Models=c(th_gt$mse, th_b_vae$mse, th_b_gan$mse),
                    Group = factor(rep(c("Baseline", "VAE Blended", "GAN Blended"), each=length(th_gt$mse))))

df_el <- data.frame(Models=c(el_gt$mse, el_b_vae$mse, el_b_gan$mse),
                    Group = factor(rep(c("Baseline", "VAE Blended", "GAN Blended"), each=length(th_gt$mse))))

ggplot(df_th, aes(x = Models, fill = Group)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 60, color = "black") +
  scale_fill_manual(values = c("blue", "red", "green", "darkred", "darkorange")) +
  labs(title = "Overlapping Histograms of MSE",
       x = "MSE", 
       y = "Frequency") +
  theme_minimal()

ggplot(df_th, aes(x = Models, color = Group)) +
  geom_density(size = 0.8, adjust = 1.5) +  # Increased size for thicker density lines
  geom_vline(data = data.frame(Group = c("Baseline", "VAE Blended", "GAN Blended"), 
                               MeanMSE = apply(th_mse, 2, mean)),
             aes(xintercept = MeanMSE, color = Group), linetype = "dashed", size = 0.8) +  # Increased size for thicker vertical lines
  scale_color_manual(values = c("Baseline" = "darkorange", "VAE"="blue", "GAN"="green", 
                                "VAE Blended" = "dodgerblue", "GAN Blended" = "limegreen")) +
  #scale_x_continuous(limits = c(0.20, 0.40), breaks = seq(0.20, 0.40, by=100)) +
  labs(title = "Density Plots of MSE for Different Datasets (Thermal context)",
       x = "MSE",
       y = "Density") +
  theme_minimal() + theme(panel.background = element_rect(fill='transparent'),
                          plot.background = element_rect(fill = 'transparent'),
                          legend.background = element_rect(fill='transparent'),
                          legend.box.background=element_rect(fill='transparent'),
                          axis.text.x = element_text(size=10, angle=0.45),
                          title = element_text(size=10))

getwd()
setwd('C:/Users/matth/OneDrive/Desktop/Fourth_Year/Final Year Project/fyp/imgs/')
ggsave(plt, filename='compare_MSE_th.pdf')
summary(df_th)

cat(mean(x), mean(y), mean(z))

# Visual comparison with box plots
th_bxplt <- cbind(th_gt$mse, th_b_vae$mse, th_b_gan$mse, th_vae$mse, th_gan$mse)
el_bxplt <- cbind(el_gt$mse, el_b_vae$mse, el_b_gan$mse, el_vae$mse, el_gan$mse)
boxplot(th_bxplt, names = c("GT", "VAE Blended", "GAN Blended", "VAE", "GAN"), 
        main = "Comparison of MSE Values",
        ylab = "MSE")
boxplot(el_bxplt, names = c("GT", "VAE Blended", "GAN Blended", "VAE", "GAN"), 
        main = "Comparison of MSE Values",
        ylab = "MSE")


th_data <- data.frame(
  MSE = c(th_gt$mse, th_b_vae$mse, th_b_gan$mse, th_vae$mse, th_gan$mse),
  Group = factor(rep(c("GT", "VAE Blended", "GAN Blended", "VAE", "GAN"), each=length(th_gt$mse)),
                 levels = c("GT", "VAE Blended", "GAN Blended", "VAE", "GAN"))
)

el_data <- data.frame(
  MSE = c(el_gt$mse, el_b_vae$mse, el_b_gan$mse, el_vae$mse, el_gan$mse),
  Group = factor(rep(c("GT", "VAE Blended", "GAN Blended", "VAE", "GAN"), each=length(el_gt$mse)),
                 levels = c("GT", "VAE Blended", "GAN Blended", "VAE", "GAN"))
)

# Boxplot with ggplot2
p <- ggplot(th_data, aes(x = Group, y = MSE, fill = Group)) +
  geom_boxplot() +
  scale_fill_manual(values = c("GT" = "darkorange", "VAE Blended" = "dodgerblue", "GAN Blended" = "limegreen",
                               "VAE" = "blue", "GAN" = "green")) +
  labs(title = "Comparison of MSE Values (Thermal Context)", y = "MSE") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = 'transparent'),
        plot.background = element_rect(fill = 'transparent'),
        legend.background = element_rect(fill = 'transparent'),
        legend.box.background = element_rect(fill = 'transparent'),
        axis.text.x = element_text(size=10, angle=45, hjust = 1))

# Save the plot with transparent background
ggsave("th_MSE_comparison.pdf", plot = p, bg = "transparent", width = 10, height = 6)

# Plot for EL data
p_el <- ggplot(el_data, aes(x = Group, y = MSE, fill = Group)) +
  geom_boxplot() +
  scale_fill_manual(values = c("GT" = "darkorange", "VAE Blended" = "dodgerblue", "GAN Blended" = "limegreen",
                               "VAE" = "blue", "GAN" = "green")) +
  labs(title = "Comparison of MSE Values (Electrical Context)", y = "MSE") +
  theme_minimal() +
  theme(panel.background = element_rect(fill = 'transparent'),
        plot.background = element_rect(fill = 'transparent'),
        legend.background = element_rect(fill = 'transparent'),
        legend.box.background = element_rect(fill = 'transparent'),
        axis.text.x = element_text(size=10, angle=45, hjust = 1))

# Save the plot for EL data
ggsave("el_MSE_comparison.pdf", plot = p_el, bg = "transparent", width = 10, height = 6)




# Perform Wilcoxon test assuming paired data
x <- el_gt$mse
y <- el_b_vae$mse
z <- el_b_gan$mse
wilcox.test(x, y, paired = FALSE)
wilcox.test(x, z, paired = FALSE)
wilcox.test(y, z, paired = FALSE)


# Assess normality for potential use of paired t-test
shapiro.test(x)
shapiro.test(y)
shapiro.test(z)

# If normality holds, perform a paired t-test
if (shapiro.test(z)$p.value > 0.05 && shapiro.test(y)$p.value > 0.05 && shapiro.test(x)$p.value > 0.05) {
  print(t.test(x, y, paired = FALSE))
  print(t.test(x, z, paired = FALSE))
}

t.test(y, x, paired=FALSE)
t.test(x, z, paried=FALSE)
t.test(z, y, paired=FALSE)

(mean(y) - mean(x))/mean(x)

cohensD(x, y)
cohensD(x, z)
cohensD(y, z)

##############

X <- cumsum(runif(10)-0.5)
t <- seq_along(X)
Y <- X + rnorm(10, 0, 0.2)
plot(t, X, "b", col="red")
points(t, Y, "b", col="blue")
var.test(X-Y, X, alternative = "less")

t.test(X-Y)

