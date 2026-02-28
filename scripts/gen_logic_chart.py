import matplotlib.pyplot as plt
import numpy as np
import os

def generate_logic_chart(output_path):
    # Set seed for reproducibility
    np.random.seed(42)
    
    # 1. Simulating Raw Data (Noisy)
    # Most are 'Walking' (0.8), but one frame randomly jumps to 'Violence' (0.75)
    raw_walking = np.array([0.85, 0.88, 0.1, 0.9, 0.86, 0.89, 0.87, 0.85, 0.88, 0.9])
    raw_violence = np.array([0.05, 0.04, 0.75, 0.02, 0.05, 0.03, 0.04, 0.06, 0.03, 0.05])
    
    # 2. Simulating Smoothed Data (Average of last 3 points for simplicity in plot)
    smoothed_walking = np.convolve(raw_walking, np.ones(3)/3, mode='same')
    smoothed_violence = np.convolve(raw_violence, np.ones(3)/3, mode='same')
    
    # Threshold line
    threshold = 0.65

    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot Raw Data
    plt.plot(raw_violence, 'r--', alpha=0.3, label='Đánh nhau (Thô - Nhiễu)')
    plt.plot(raw_walking, 'g--', alpha=0.3, label='Đi bộ (Thô)')

    # Plot Smoothed Data
    plt.plot(smoothed_violence, 'r-', linewidth=3, label='Đánh nhau (Đã làm mịn)')
    plt.plot(smoothed_walking, 'g-', linewidth=3, label='Đi bộ (Đã làm mịn)')

    # Plot Threshold
    plt.axhline(y=threshold, color='blue', linestyle=':', linewidth=2, label='Ngưỡng tin cậy (65%)')

    # Annotations
    plt.annotate('Nhiễu ngẫu nhiên\n(Bị triệt tiêu)', xy=(2, 0.75), xytext=(3, 0.75),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    plt.annotate('Dưới ngưỡng 65%\n-> Bị loại bỏ (Safe)', xy=(2, 0.28), xytext=(3, 0.35),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5), color='red')

    plt.title('Minh họa Thuật toán Làm mịn & Ngưỡng tin cậy', fontsize=16, fontweight='bold')
    plt.xlabel('Thời gian (Inference Steps)', fontsize=12)
    plt.ylabel('Xác suất (Confidence)', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Save to artifacts directory
    target_dir = r"C:\Users\Phuong Nam\.gemini\antigravity\brain\cc0f0a8e-0b7c-4fdd-b383-ff04e5f52dff"
    output_img = os.path.join(target_dir, "smoothing_logic_chart.png")
    generate_logic_chart(output_img)
    print(f"Chart generated at: {output_img}")
