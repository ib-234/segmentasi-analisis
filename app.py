import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Segmentasi Pelanggan GoFood", layout="centered")

# --- Inisialisasi Session State ---
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "data" not in st.session_state:
    st.session_state.data = None

# --- Halaman Upload File ---
if st.session_state.page == "upload":
    st.caption("Halaman 1 dari 3")
    st.title("📁 1. Upload File")
    uploaded_file = st.file_uploader("Unggah file CSV", type=None)  # type=None biar bisa semua format

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success("File berhasil diunggah!")
            st.dataframe(df.head())

            if st.button("➡️ Lanjut ke Pra-pemrosesan"):
                st.session_state.page = "preprocessing"
                st.experimental_rerun()
        else:
            st.toast("⚠️ Format file tidak sesuai! Harap unggah file dengan format CSV.", icon="🚫")

# --- Halaman Pra-pemrosesan & Clustering ---
elif st.session_state.page == "preprocessing":
    st.caption("Halaman 2 dari 3")
    st.title("🔤 2. Pra-pemrosesan & Clustering")

    if st.session_state.data is None:
        st.warning("Silakan unggah file terlebih dahulu di halaman upload.")
        if st.button("⬅️ Kembali ke Upload"):
            st.session_state.page = "upload"
            st.experimental_rerun()
    else:
        df = st.session_state.data
        df_encoded = df.copy()

        # Label Encoding (mulai dari 1, bukan 0)
        label_encoders = {}
        for col in df_encoded.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col]) + 1
            label_encoders[col] = le

        st.subheader("Data Setelah Encoding (mulai dari 1)")
        st.dataframe(df_encoded.head())

        # Pilih kolom numerik
        numeric_cols = df_encoded.select_dtypes(include=["int64", "float64"]).columns.tolist()
        selected_cols = st.multiselect("Pilih kolom numerik untuk clustering:", numeric_cols, default=numeric_cols)

        if len(selected_cols) >= 2:
            data_cluster = df_encoded[selected_cols]

            # Elbow Method
            st.subheader("📈 Elbow Method")
            K = range(1, 11)
            inertia = []
            for k in K:
                model = KMeans(n_clusters=k, random_state=42)
                model.fit(data_cluster)
                inertia.append(model.inertia_)

            fig, ax = plt.subplots()
            ax.plot(K, inertia, marker='o')
            ax.set_xlabel('Jumlah Klaster (K)')
            ax.set_ylabel('Inersia')
            ax.set_title('Elbow Method')
            st.pyplot(fig)

            # Slider jumlah cluster
            selected_k = st.slider("Pilih Jumlah Klaster", 2, 10, 3)

            # KMeans
            kmeans = KMeans(n_clusters=selected_k, random_state=42)
            labels = kmeans.fit_predict(data_cluster)
            data_cluster['Cluster'] = labels + 1

            st.subheader("📋 Hasil Klastering")
            st.dataframe(data_cluster)

            # Distribusi cluster
            fig2, ax2 = plt.subplots()
            sns.countplot(x='Cluster', data=data_cluster, palette='Set2', ax=ax2)
            ax2.set_title("Distribusi Data per Klaster")
            st.pyplot(fig2)

            # Silhouette Score
            score = silhouette_score(data_cluster.drop("Cluster", axis=1), data_cluster["Cluster"])
            st.markdown(f"**Silhouette Score:** `{score:.3f}`")

            # Visualisasi PCA
            st.subheader("🌀 Visualisasi Hasil Klastering (PCA 2D)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data_cluster.drop("Cluster", axis=1))
            pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
            pca_df["Cluster"] = data_cluster["Cluster"]

            fig3, ax3 = plt.subplots()
            sns.scatterplot(x="PC1", y="PC2", hue="Cluster", data=pca_df, palette='Set2', s=80, ax=ax3)
            ax3.set_title("Visualisasi Klastering dengan PCA")
            st.pyplot(fig3)

            # --- Tambahan: Persentase per Klaster ---
            st.subheader("📊 Persentase Data per Klaster")
            cluster_counts = data_cluster["Cluster"].value_counts().sort_index()
            cluster_percent = (cluster_counts / len(data_cluster) * 100).round(2)

            summary_df = pd.DataFrame({
                "Cluster": cluster_counts.index,
                "Jumlah Data": cluster_counts.values,
                "Persentase (%)": cluster_percent.values
            })

            st.dataframe(summary_df)

            # Pie Chart
            fig4, ax4 = plt.subplots()
            ax4.pie(cluster_percent, labels=[f"Cluster {i}" for i in cluster_counts.index],
                    autopct="%1.1f%%", colors=sns.color_palette("Set2", len(cluster_counts)))
            ax4.set_title("Proporsi Data per Klaster")
            st.pyplot(fig4)

            st.markdown(f"""
            **Kesimpulan PCA & Distribusi:**
            Visualisasi PCA menunjukkan pola klaster dengan jumlah klaster **{selected_k}**.
            Dari distribusi, terlihat proporsi data per klaster cukup jelas. Hal ini membantu
            memahami segmentasi pelanggan berdasarkan kesamaan atribut, di mana setiap klaster
            memiliki persentase kontribusi berbeda terhadap keseluruhan data.
            """)

        else:
            st.warning("Pilih minimal dua kolom numerik untuk melakukan clustering.")

        st.markdown("---")
        if st.button("⬅️ Kembali ke Upload"):
            st.session_state.page = "upload"
            st.experimental_rerun()

        if st.button("🚪 Exit"):
            st.session_state.page = "exit"
            st.experimental_rerun()

# --- Halaman Exit ---
elif st.session_state.page == "exit":
    st.caption("Halaman 3 dari 3")
    st.markdown(
        "<h3 style='text-align: center;'>Terima kasih telah menggunakan website ini, semoga bermanfaat 🙏</h3>",
        unsafe_allow_html=True
    )
    if st.button("⬅️ Kembali ke Upload"):
        st.session_state.page = "upload"
        st.experimental_rerun()
