import numpy as np
import pandas as pd
import math
import os

import plotly.graph_objects as go
import matplotlib.pyplot as plt


from PIL import Image as PilImage


# ===== IMAGE EXPLORATION =====
def explore_image_files(file_paths, explore_values=False):
    ''' Open each image to extract extra info'''
    # Define the variables of smalles and biggest images
    image_sizes = []
    color_channels = []
    formats = []
    min_vals = []
    max_vals = []
    ratios = []

    # Iteration loop for each folder to compare the image sizes

    for file_path in file_paths:
        with PilImage.open(file_path) as img:
            image_sizes.append(img.size)
            color_channels.append(img.mode)
            formats.append(img.format)
            ratios.append(img.size[0]/img.size[1])

            if explore_values:
                # Convert to numpy to check the actual data type, Takes alot of time
                img_array = np.array(img)

                # Value range
                min_vals.append(img_array.min())
                max_vals.append(img_array.max())

    if explore_values:
        return image_sizes, color_channels, formats, ratios, min_vals, max_vals
    else:
        return image_sizes, color_channels, formats, ratios
    

def smallest_largest_image(metadata, directory):
    
    # Find largest and smallest images by pixel count
    largest = metadata.loc[metadata["pixel_count"].idxmax()]
    smallest = metadata.loc[metadata["pixel_count"].idxmin()]

    largest_image_path = directory + largest["file_path"]
    largest_pixels = largest["pixel_count"]
    largest_dims = (largest["width"], largest["height"])

    smallest_image_path = directory + smallest["file_path"]
    smallest_pixels = smallest["pixel_count"]
    smallest_dims = (smallest["width"], smallest["height"])

    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Largest image
    img_largest = PilImage.open(largest_image_path)
    axes[0].imshow(img_largest)
    axes[0].axis('off')
    axes[0].set_title(
        f"LARGEST IMAGE\n{os.path.basename(largest_image_path)}\n"
        f"{largest_dims[0]} x {largest_dims[1]} pixels\n"
        f"({largest_pixels:,} total pixels)",
        fontsize=12, fontweight='bold'
    )

    # Smallest image
    img_smallest = PilImage.open(smallest_image_path)
    axes[1].imshow(img_smallest)
    axes[1].axis('off')
    axes[1].set_title(
        f"SMALLEST IMAGE\n{os.path.basename(smallest_image_path)}\n"
        f"{smallest_dims[0]} x {smallest_dims[1]} pixels\n"
        f"({smallest_pixels:,} total pixels)",
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    plt.show()

    
    print(f"Largest image path:  {largest_image_path}")
    print(f"\nSmallest image path: {smallest_image_path}")



# ===== DATA PREPROCESSING =====
def get_distances(metadata, centroids):
    # Calculate distances to centroids
    distances = []

    for idx, row in metadata.iterrows():
        family = row['family']
        if family in centroids:
            # Calculate Euclidean distance
            distance = np.linalg.norm(row['embedding'] - centroids[family])
            distances.append(distance)
        else:
            distances.append(np.nan)

    return distances


def distance_to_centroid(train_df, test_df):
    # Calculate a centroid for each family based on the embedings from the train_df
    centroids = {}

    for family in train_df['family'].unique():
        # Get all the embedings for 1 family
        family_embeddings = train_df[train_df['family'] == family]['embedding'].tolist()

        if len(family_embeddings) > 0:
            # Stack embeddings and calculate mean
            embeddings_array = np.stack(family_embeddings)
            centroids[family] = np.mean(embeddings_array, axis=0)
    
    # Return the distances for both train and test based on the centroids of the train
    return get_distances(train_df, centroids), get_distances(test_df, centroids)












# ===== VISUALIZATIONS =====
# -----      PLOTS     -----
def plot_phylum_counts(metadata):
    # Get phylum counts
    phylum_counts = metadata['phylum'].value_counts()

    # Calculate number of families per phylum
    families_per_phylum = metadata.groupby('phylum')['family'].nunique()

    # Create custom hover data with family counts
    hover_data = []
    for phylum in phylum_counts.index:
        image_count = phylum_counts[phylum]
        family_count = families_per_phylum[phylum]
        percentage = (image_count / len(metadata)) * 100
        hover_data.append([image_count, percentage, family_count])

    # Convert to numpy array for easy indexing
    hover_data = np.array(hover_data)

    # Create figure
    fig = go.Figure(go.Bar(
        x=phylum_counts.index,
        y=phylum_counts.values,
        marker=dict(
            color='#6366f1',
            line=dict(color='#4f46e5', width=0.5)
        ),
        text=phylum_counts.values,
        textposition='outside',
        textfont=dict(size=12),
        customdata=hover_data,
        hovertemplate='<b>%{x}</b><br>' +
                    'Images: <b>%{y}</b> (<b>%{customdata[1]:.2f}%</b>)<br>' +
                    'Families: %{customdata[2]}<br>'
        
    ))

    fig.update_layout(
        title={
            'text': '<b>Distribution of Species by Phylum</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Phylum',
        yaxis_title='Count',
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(size=11),
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),
        font=dict(size=11),
        showlegend=False
    )

    fig.show()


def plot_family_counts(metadata):
    # Get all family counts
    family_counts = metadata['family'].value_counts()

    # Create figure with scrollable y-axis
    fig = go.Figure(go.Bar(
        x=family_counts.values,
        y=family_counts.index,
        orientation='h',
        marker=dict(
            color='#6366f1',
            line=dict(color='#4f46e5', width=0.5)
        ),
        text=family_counts.values,
        textposition='outside',
        textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>Images: <b>%{x}</b> (<b>%{customdata:.2f}%</b>)<extra></extra>',
        customdata=(family_counts.values / len(metadata)) * 100
    ))

    fig.update_layout(
        title={
            'text': '<b>Distribution Of Species By Family</b><br>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Number of Images',
        yaxis_title='Family',
        height=max(1000, len(family_counts) * 15),  # Dynamic height based on number of families
        width=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        yaxis=dict(
            autorange='reversed',  # Highest count on top
            tickfont=dict(size=9),
            showgrid=False
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),
        font=dict(size=11),
        margin=dict(l=200, r=100, t=100, b=50),  # More space for family names
        showlegend=False
    )

    fig.show()


def plot_color_chanels_counts(metadata):
    
    # Plot colour channel distribution
    color_channel_mapping = {
        'L': 'Greyscale',
        'RGB': 'RGB',
        'RGBA': 'RGBA',
        'P': 'Palette',
        'CMYK': 'CMYK',
        '1': 'Binary',
        'LA': 'Greyscale + Alpha'
    }

    # Get value counts and map to readable names
    color_counts = metadata['color_channel'].value_counts()
    color_counts.index = color_counts.index.map(lambda x: color_channel_mapping.get(x, x))

    # Create figure
    fig = go.Figure(go.Bar(
        x=color_counts.index,
        y=color_counts.values,
        marker=dict(
            color='#6366f1',
            line=dict(color='#4f46e5', width=0.5)
        ),
        text=color_counts.values,
        textposition='outside',
        textfont=dict(size=12),
        hovertemplate='<b>%{x}</b><br>Images: %{y}<br>Percentage: %{customdata:.2f}%<extra></extra>',
        customdata=(color_counts.values / len(metadata)) * 100
    ))

    fig.update_layout(
        title={
            'text': '<b>Distribution Of Color Channels</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Color Channel',
        yaxis_title='Count',
        height=600,
        width=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickfont=dict(size=11),
            showgrid=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),
        font=dict(size=11),
        showlegend=False
    )

    fig.show()


def plot_image_size_scatter(metadata, common_ratio, width_max):
    # Create figure with custom styling
    fig = go.Figure()

    # Add scatter plot with better styling
    fig.add_trace(go.Scatter(
        x=metadata['width'],
        y=metadata['height'],
        mode='markers',
        marker=dict(
            size=4,
            color='#6366f1',
            opacity=0.4,
            line=dict(width=0)
        ),
        name='Images',
        hovertemplate='<b>Width:</b> %{x}px<br><b>Height:</b> %{y}px<extra></extra>'
    ))

    # Add aspect ratio reference line (4:3)
    max_y = metadata['height'].max()
    max_x_for_ratio = common_ratio * max_y
    fig.add_trace(go.Scatter(
        x=[0, max_x_for_ratio],
        y=[0, max_y],
        mode='lines',
        line=dict(color='#ef4444', width=2.5, dash='dash'),
        name='4:3 Aspect Ratio',
        hoverinfo='skip'
    ))

    # Add width threshold line
    fig.add_trace(go.Scatter(
        x=[width_max, width_max],
        y=[0, max_y],
        mode='lines',
        line=dict(color='#10b981', width=2.5, dash='dash'),
        name=f'Max Width ({width_max}px)',
        hoverinfo='skip'
    ))

    # Update layout with legend on the right and bold title
    fig.update_layout(
        title={
            'text': '<b>Image Size Distribution Analysis</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#1f2937'}
        },
        xaxis=dict(
            title='Width (pixels)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=14, color='#374151')
        ),
        yaxis=dict(
            title='Height (pixels)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            zeroline=False,
            title_font=dict(size=14, color='#374151')
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=700,
        showlegend=True,
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#d1d5db',
            borderwidth=1,
            font=dict(size=11)
        ),
        hovermode='closest'
    )

    fig.show()


def plot_model_results(history, metric):

    # Extract loss values
    train_metric = history.history[metric]
    val_metric = history.history[f'val_{metric}']
    epochs = np.arange(1, len(train_metric) + 1)

    # Build figure
    fig = go.Figure()

    # ---- Train Loss ----
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_metric,
        mode='lines',
        name='Train',
        line=dict(width=2, color='#6366f1'),
        hovertemplate='Train ' + metric.title() + ': <b>%{y:.4f}</b><extra></extra>'
    ))

    # ---- Validation Loss ----
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_metric,
        mode='lines',
        name='Validation',
        line=dict(width=2, color='#ffa500'),
        hovertemplate='Validation ' + metric.title() + ': <b>%{y:.4f}</b><extra></extra>'
    ))

    # ---- Layout ----
    fig.update_layout(
        hovermode='x unified',
        title={
            'text': f'<b>Model {metric.title()}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='Epochs',
        yaxis_title=metric.title(),
        height=600,
        width=1000,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=11),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1
        ),

        legend=dict(
            orientation='h',
            x=0.5,
            xanchor='center',
            y=1.05,
            font=dict(size=12)
        )
    )

    fig.show()


# -----     IMAGES     -----

def plot_images_from_directory(directory, img_per_class=5):
    # Print 5 example images of each class (+-3min)
    for label in os.listdir(directory):
        path = directory + str(label)

        if not os.path.isdir(path):
            print(f"Directory {path} does not exist.")
            continue

        folder_data = os.listdir(path)
        k = 0
        print(f'{label} ({len(folder_data)} images)')

        # Collect image paths
        image_paths = []
        for image_path in folder_data:
            if k < img_per_class:                                               
                full_path = os.path.join(path, image_path)
                image_paths.append(full_path)
                k += 1

        # Display images
        if image_paths:
            fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 3))
            if len(image_paths) == 1:
                axes = [axes]

            for ax, img_path in zip(axes, image_paths):
                img = PilImage.open(img_path)
                ax.imshow(img)
                ax.axis('off')

            plt.tight_layout()
            plt.show()


def plot_images_from_generator(generator, num_batches=False):

    # Get class names from the generator
    class_names = list(generator.class_indices.keys())

    if not num_batches:
        # If number of batches not set, plot all
        num_batches = int(math.ceil(generator.n / generator.batch_size))

    for i in range(num_batches):
        images, labels = next(generator)

        # Plot the images in the current batch
        batch_size_actual = images.shape[0]
        n_cols = min(8, batch_size_actual)                  # <-- set max columns per row
        n_rows = math.ceil(batch_size_actual / n_cols)

        plt.figure(figsize=(3 * n_cols, 3 * n_rows))

        for j in range(batch_size_actual):
            ax = plt.subplot(n_rows, n_cols, j + 1)
            plt.imshow(images[j])

            # Get the index of the highest probability to find the class name
            label_idx = np.argmax(labels[j])
            plt.title(class_names[label_idx], fontsize=9)
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def plot_furthest_images_batched(directory, metadata, N, n_cols=10):
    # Get the images sorted by distance to centroid (make a copy to not mess up the original metadata)
    sorted_metadata = metadata.sort_values(by='distance_to_centroid', ascending=False).copy()

    for i in range(0, N, n_cols):
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2, 2))

        # Flatten axes to simplify indexing
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx in range(i, min(i + n_cols, N)):
            if idx < len(sorted_metadata):
                # Show the image with family and ID in the title
                img_path = os.path.join(directory, sorted_metadata.iloc[idx]['file_path'])
                img = PilImage.open(img_path)
                axes[idx - i].imshow(img)
                family = sorted_metadata.iloc[idx]['family']
                axes[idx - i].set_title(f'{family} (id={idx})', fontsize=8)
                axes[idx - i].axis('off')

            else:
                # Hide the subplot if there is no image
                axes[idx - i].axis('off')

        plt.tight_layout()
        plt.show()


def plot_good_vs_bad_images(good_images_df, outliers, directory, good_images=2):

    for family in outliers['family'].unique():
        # Get all the image paths sorted
        image_paths = list(directory+good_images_df[good_images_df.family == family].sort_values('distance_to_centroid')['file_path'])[:good_images]
        image_paths.extend(list(directory+outliers[outliers.family == family].sort_values('distance_to_centroid', ascending=False)['file_path']))

        if image_paths:
            # Set number of columns and calculate rows needed
            n_cols = 7
            n_images = len(image_paths)
            n_rows = (n_images + n_cols - 1) // n_cols 

            # Create subplots with appropriate dimensions
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows))

            # Flatten axes array for easier iteration
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # Display images
            for idx, img_path in enumerate(image_paths):
                img = PilImage.open(img_path).resize((224, 224), PilImage.LANCZOS)
                axes[idx].imshow(img)
                axes[idx].axis('off')

            # Add green outline for good images (first good_count images)
                if idx < good_images:
                    rect = plt.Rectangle((0, 0), 223, 223, fill=False,
                                        edgecolor='lime', linewidth=3)
                    axes[idx].add_patch(rect)

            # Hide any unused subplots
            for idx in range(n_images, n_rows * n_cols):
                axes[idx].axis('off')

            plt.suptitle(f"{family}, {good_images_df[good_images_df.family == family].shape[0]} good images left")
            plt.tight_layout()
            plt.show()



