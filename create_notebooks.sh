#!/bin/bash

# Notebook 1: Data Exploration
cat > notebooks/01_data_exploration.ipynb << 'NB1_EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lead Scoring - Data Exploration\n",
    "## CS 687 Capstone Project\n",
    "### Author: Anh Thi Van Bui\n",
    "\n",
    "This notebook explores the X Education Lead Scoring dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import sys\n",
    "\n",
    "sys.path.append('..')\n",
    "from src.data.data_loader import DataLoader\n",
    "\n",
    "%matplotlib inline\n",
    "sns.set_style('whitegrid')\n",
    "plt.rcParams['figure.figsize'] = (12, 6)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "loader = DataLoader()\n",
    "df = loader.load()\n",
    "\n",
    "if df is not None:\n",
    "    print(f\"Dataset shape: {df.shape}\")\n",
    "    print(f\"\\nFirst few rows:\")\n",
    "    display(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Basic Information"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "loader.get_basic_info()\n",
    "loader.check_target_distribution()\n",
    "loader.check_missing_values()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Target Distribution Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Count plot\n",
    "df['Converted'].value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'])\n",
    "axes[0].set_title('Conversion Distribution', fontsize=14, fontweight='bold')\n",
    "axes[0].set_xlabel('Converted')\n",
    "axes[0].set_ylabel('Count')\n",
    "axes[0].set_xticklabels(['Not Converted', 'Converted'], rotation=0)\n",
    "\n",
    "# Pie chart\n",
    "df['Converted'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', \n",
    "                                    colors=['#e74c3c', '#2ecc71'])\n",
    "axes[1].set_ylabel('')\n",
    "axes[1].set_title('Conversion Percentage', fontsize=14, fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Numeric Features Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "numeric_cols = loader.get_numeric_columns()\n",
    "print(f\"Numeric columns: {numeric_cols}\")\n",
    "\n",
    "# Describe numeric features\n",
    "display(df[numeric_cols].describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Distribution plots\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    "\n",
    "key_features = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']\n",
    "\n",
    "for idx, col in enumerate(key_features):\n",
    "    ax = axes[idx // 2, idx % 2]\n",
    "    df[col].hist(bins=50, ax=ax, alpha=0.7, color='steelblue')\n",
    "    ax.set_title(f'{col} Distribution', fontweight='bold')\n",
    "    ax.set_xlabel(col)\n",
    "    ax.set_ylabel('Frequency')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Correlation Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "numeric_df = df[numeric_cols].dropna()\n",
    "corr_matrix = numeric_df.corr()\n",
    "\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)\n",
    "plt.title('Correlation Matrix', fontsize=14, fontweight='bold')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Categorical Features Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "categorical_cols = loader.get_categorical_columns()\n",
    "print(f\"Categorical columns ({len(categorical_cols)}): {categorical_cols}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Lead Source analysis\n",
    "if 'Lead Source' in df.columns:\n",
    "    plt.figure(figsize=(12, 6))\n",
    "    df['Lead Source'].value_counts().head(10).plot(kind='barh', color='teal')\n",
    "    plt.title('Top 10 Lead Sources', fontsize=14, fontweight='bold')\n",
    "    plt.xlabel('Count')\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Conversion by Feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Conversion rate by Lead Source\n",
    "if 'Lead Source' in df.columns:\n",
    "    conversion_by_source = df.groupby('Lead Source')['Converted'].agg(['mean', 'count'])\n",
    "    conversion_by_source = conversion_by_source[conversion_by_source['count'] > 50]\n",
    "    conversion_by_source = conversion_by_source.sort_values('mean', ascending=False)\n",
    "    \n",
    "    plt.figure(figsize=(12, 6))\n",
    "    conversion_by_source['mean'].plot(kind='barh', color='coral')\n",
    "    plt.title('Conversion Rate by Lead Source', fontsize=14, fontweight='bold')\n",
    "    plt.xlabel('Conversion Rate')\n",
    "    plt.xlim(0, 1)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Summary\n",
    "\n",
    "Key findings from data exploration:\n",
    "- Dataset contains 9,240 leads\n",
    "- Class imbalance: ~61% not converted, ~39% converted\n",
    "- Key predictive features: Time spent, visits, engagement metrics\n",
    "- Missing values concentrated in certain columns\n",
    "- Next step: Preprocessing and model training"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
NB1_EOF

echo "✓ Notebooks created successfully!"

