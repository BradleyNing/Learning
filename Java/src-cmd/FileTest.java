class FileTest {
	public static void main(String[] args) {
		save();
	}

	public void save() {
		String data = "Data to save";
		FileOutputStream out = null;
		BufferedWriter writer = null;

		try {
			out = openFileOutput("data.txt");
			writer = new BufferedWriter(new OutputStreamWriter(out));
			writer.writer(data);
		} catch(Exception e) {
			System.out.println(e);
		} finally {
			writer.close();
		}
	}
}
