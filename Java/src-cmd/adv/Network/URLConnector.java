import java.net.*;
import java.io.*;

public class URLConnector {
	public static void main(String[] args) {
		try {
			URL cs = new URL("http://www.google.com");
			URLConnection tc = cs.openConnection();
			BufferedReader in  = new BufferedReader(new InputStreamReader(tc.getInputStream()));
			String inputLine;
			System.out.println(in);
			while((inputLine = in.readLine()) != null) {
				System.out.println(inputLine);
			}

			in.close();
		} catch(Exception e) {System.out.println(e);}
	}

	public static String sendGet(String url, String param) {
		String result = "";
		BufferedReader in = null;
		try {
			String urlNameString = url + "?" + param;
			URL realUrl = new URL(urlNameString);
			URLConnection connection = realUrl.openConnection();
			connection.setRequestProperty("accept", "*/*");
			connection.setRequestProperty("connection", "Keep-Alive");
			connection.connect();

			in = new BufferedReader(new InputStreamReader connection.getInputStream());
			String line;
			while ((line =in.readLine()) != null) {
				result += line;
			}
		}catch(Exception e) {System.out.println(e);}
		finally {
			try {
				if (in != null) in.close();
			} catch (Exception e) {;}
		}
		return result;
	}

	public static String sendPost(String url, String param) {
		PrintWriter out = null;
		BufferedReader in = null;
		String result = "";
		try {
			URL realUrl = new URL(url);
			URLConnection conn = realUrl.openConnection();
			conn.setRequestProperty("accept", "*/*");
			conn.setRequestProperty("connection", "Keep-Alive");
			conn.setDoOutput(true);
			out = new PrintWriter(conn.getOutputStream());
			out.print(param);
			out.flush();

			in = new BufferedReader(new InputStreamReader connection.getInputStream());
			String line;
			while ((line =in.readLine()) != null) {
				result += line;
			}
		}catch(Exception e) {System.out.println(e);}
		finally {
			try {
				if (in != null) in.close();
				if (out != null) out.close();
			} catch (Exception e) {;}
		}
		return result;
	}
}

// HttpURLConnection